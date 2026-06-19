"""CDP-over-Playwright proxy — the Phase-5 BiDi bring-up shortcut.

browser-use's watchdogs + ``dom/service`` talk to the browser through
~731 raw ``session.cdp_client.send.<Domain>.<Method>(...)`` call sites.
Porting each one to a BiDi/Playwright branch is a multi-week,
high-churn edit that fights every future upstream rebase.

This module takes the cheaper, single-chokepoint route: a **CDP facade**
that exposes the *exact* ``cdp_use.CDPClient`` surface (``.send`` /
``.register`` / ``.send_raw``) but, instead of marshalling JSON-RPC over
a WebSocket to Chrome, **translates each CDP method into Playwright calls**
against a Firefox/Camoufox page reached over BiDi.

Why this is a drop-in (not a reimplementation of the typed surface):
``cdp_use``'s ``CDPLibrary`` / ``CDPRegistrationLibrary`` are generated
shims whose every method bottoms out in ``client.send_raw(method=...,
params=..., session_id=...)`` (see ``cdp_use/cdp/page/library.py``). So
we reuse those generated libraries verbatim and only re-implement
``send_raw`` — the whole ``.send.Domain.Method`` tree then routes through
our translation table for free, with guaranteed signature fidelity.

731 call sites collapse to **79 distinct CDP methods**; on the BiDi
happy-path (non-essential watchdogs disabled — downloads/popups/network/
fetch/emulation/screencast) the live surface is ~20. We implement those
on demand, harness-driven: run ``Agent.run()`` → it raises
``CdpMethodNotImplemented`` naming the next method → implement → repeat.

Architecture note: this is the **in-process facade** (stage 1). The same
``_HANDLERS`` translation core can later be wrapped in a real CDP
WebSocket *server* (stage 2) for a zero-fork-edit sidecar deploy —
only the transport shell (ws framing + Target/session multiplexing)
differs; the translation logic is shared.

The one genuinely hard domain is **DOM identity** (``DOM.resolveNode`` /
``getBoxModel`` / ``DOMSnapshot.captureSnapshot`` / ``Accessibility`` —
everything keyed on ``backendNodeId`` / ``objectId``). Playwright exposes
no such ids, so :class:`NodeRegistry` maintains a *synthetic* node table
(built from an a11y/JS snapshot, each id remembering a JSHandle/locator).
That part is a reimplementation no matter the strategy; the facade merely
confines it here and keeps ``dom/service.py`` unmodified. It is the next
milestone — methods in that domain currently raise NotImplemented.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from cdp_use import CDPClient

if TYPE_CHECKING:
	from playwright.async_api import Page as PlaywrightPage

	from browser_use.browser.connection import BidiBrowserConnection


class CdpMethodNotImplemented(NotImplementedError):
	"""Raised by :meth:`BidiCdpProxy.send_raw` for a CDP method that has no
	Playwright translation yet. The message names the method so the
	harness loop tells you exactly what to implement next."""


# A single synthetic target/session — BidiBrowserConnection is currently
# single-context/single-page. When multi-tab BiDi tracking lands these
# become a small registry keyed by Playwright page identity.
_SYNTHETIC_TARGET_ID = 'BIDI-TARGET-0'
_SYNTHETIC_SESSION_ID = 'BIDI-SESSION-0'
_SYNTHETIC_FRAME_ID = 'BIDI-FRAME-0'
_SYNTHETIC_BROWSER_CONTEXT_ID = 'BIDI-BROWSERCTX-0'


class NodeRegistry:
	"""Synthetic CDP node-identity table over Playwright handles.

	CDP traffics in integer ``backendNodeId`` and string ``objectId``;
	Playwright exposes neither. This registry mints monotonic ids and
	remembers, per id, the Playwright ``JSHandle`` / ``ElementHandle`` (or
	a resolvable CSS path) so that later ``DOM.*`` / ``Runtime.callFunctionOn``
	calls keyed on those ids can be answered against the live page.

	NOTE: stub. Population (from ``DOM.getDocument`` /
	``DOMSnapshot.captureSnapshot`` / a11y snapshot) and resolution are the
	OBSERVE milestone — see module docstring §"DOM identity".
	"""

	def __init__(self) -> None:
		self._backend_node_seq = itertools.count(1)
		self._object_seq = itertools.count(1)
		# backendNodeId -> opaque handle/selector payload
		self._by_backend_node: dict[int, Any] = {}
		# objectId -> opaque handle payload
		self._by_object: dict[str, Any] = {}

	def mint_backend_node(self, handle: Any) -> int:
		nid = next(self._backend_node_seq)
		self._by_backend_node[nid] = handle
		return nid

	def mint_object(self, handle: Any) -> str:
		oid = f'BIDI-OBJ-{next(self._object_seq)}'
		self._by_object[oid] = handle
		return oid

	def handle_for_backend_node(self, backend_node_id: int) -> Any:
		return self._by_backend_node.get(backend_node_id)

	def handle_for_object(self, object_id: str) -> Any:
		return self._by_object.get(object_id)

	def clear(self) -> None:
		self._by_backend_node.clear()
		self._by_object.clear()


class _FakeWs:
	"""Minimal stand-in so liveness checks that poke ``client.ws.state``
	(e.g. ``BrowserSession._is_connected``) see an OPEN-looking channel.
	The real liveness source of truth on BiDi is
	``BidiBrowserConnection.is_open`` (Playwright ``browser.is_connected()``)."""

	def __init__(self, conn: 'BidiBrowserConnection') -> None:
		self._conn = conn

	@property
	def state(self) -> str:
		try:
			return 'OPEN' if self._conn.is_open else 'CLOSED'
		except Exception:
			return 'CLOSED'


class BidiCdpProxy(CDPClient):
	"""Drop-in for :class:`cdp_use.CDPClient`, backed by Playwright/BiDi.

	Subclasses ``CDPClient`` (but does NOT call its ``__init__`` — that would
	open a real WebSocket) purely so ``isinstance(proxy, CDPClient)`` holds:
	``CDPSession.cdp_client`` is a Pydantic field typed ``CDPClient`` and is
	validated by isinstance. We re-create the same public attributes
	(``send`` / ``register`` / ``_event_registry``) ourselves.

	Exposes the same ``.send`` / ``.register`` / ``.send_raw`` /
	``.start`` / ``.stop`` surface so it can be returned from
	``BrowserSession.cdp_client`` unchanged. All command dispatch routes
	through :meth:`send_raw` → :attr:`_HANDLERS`.
	"""

	def __init__(self, connection: 'BidiBrowserConnection') -> None:
		# NB: intentionally NOT calling super().__init__() — CDPClient's
		# constructor would try to stand up a websocket client. We only need
		# the type identity + the cdp_use typed libraries (built below).
		self._conn = connection
		self.ws = _FakeWs(connection)
		self.msg_id = 0

		# Reuse cdp_use's generated typed libraries verbatim — they bottom
		# out in self.send_raw(), so the entire .send.Domain.Method tree is
		# ours for free, with exact signatures.
		from cdp_use.cdp.library import CDPLibrary
		from cdp_use.cdp.registration_library import CDPRegistrationLibrary
		from cdp_use.cdp.registry import EventRegistry

		self.send = CDPLibrary(self)
		self._event_registry = EventRegistry()
		self.register = CDPRegistrationLibrary(self._event_registry)

		self._nodes = NodeRegistry()
		self._started = False
		# Cached single-pass DOM scan (tree + flattened snapshot), keyed by URL.
		self._scan_cache: Optional[dict] = None
		self._scan_url: Optional[str] = None

	# ── page access ─────────────────────────────────────────────────────
	@property
	def _page(self) -> 'PlaywrightPage':
		"""The active Playwright page. Single-tab posture for now."""
		return self._conn.current_page

	def _adapter(self) -> Any:
		from browser_use.browser.adapter import PlaywrightBrowserAdapter

		return PlaywrightBrowserAdapter(self._page)

	# ── CDPClient lifecycle parity ──────────────────────────────────────
	async def start(self) -> None:
		# The underlying BiDi channel is owned/started by BidiBrowserConnection.
		self._started = True

	async def stop(self) -> None:
		self._started = False
		self._nodes.clear()

	async def emit_event(self, method: str, params: Any = None, session_id: Optional[str] = None) -> bool:
		"""Push a synthetic CDP event into the registry (used to translate
		Playwright events — framenavigated, dialog, etc. — into the CDP
		events watchdogs subscribed to via ``.register``)."""
		return await self._event_registry.handle_event(method, params or {}, session_id)

	# ── the one method that matters: CDP → Playwright ───────────────────
	async def send_raw(
		self,
		method: str,
		params: Optional[Any] = None,
		session_id: Optional[str] = None,
	) -> dict[str, Any]:
		handler = self._HANDLERS.get(method)
		if handler is None:
			raise CdpMethodNotImplemented(
				f'BidiCdpProxy: no Playwright translation for CDP method {method!r} yet. '
				f'Implement it in cdp_proxy._HANDLERS (params={params!r}).'
			)
		return await handler(self, params or {}, session_id)

	# ════════════════════════════════════════════════════════════════════
	# Handlers. Each: async (self, params: dict, session_id) -> dict (CDP-shaped result).
	# Grouped by domain. Keep translations faithful to the fields browser_use READS.
	# ════════════════════════════════════════════════════════════════════

	# ── no-ops: enable/disable + observability domains that have no BiDi
	#    analog. Returning {} keeps the call sites happy without effect. ──
	async def _noop(self, params: dict, session_id: Optional[str]) -> dict:
		return {}

	# ── Browser ─────────────────────────────────────────────────────────
	async def _Browser_getVersion(self, params: dict, session_id: Optional[str]) -> dict:
		ua = ''
		try:
			ua = await self._page.evaluate('navigator.userAgent')
		except Exception:
			pass
		return {
			'protocolVersion': '1.3',
			'product': 'Firefox/Camoufox(BiDi)',
			'revision': 'bidi-proxy',
			'userAgent': ua,
			'jsVersion': '',
		}

	# ── Target ──────────────────────────────────────────────────────────
	def _target_info(self, url: str = 'about:blank') -> dict:
		return {
			'targetId': _SYNTHETIC_TARGET_ID,
			'type': 'page',
			'title': '',
			'url': url,
			'attached': True,
			'browserContextId': _SYNTHETIC_BROWSER_CONTEXT_ID,
		}

	async def _Target_getTargets(self, params: dict, session_id: Optional[str]) -> dict:
		url = ''
		try:
			url = self._page.url
		except Exception:
			pass
		return {'targetInfos': [self._target_info(url)]}

	async def _Target_createTarget(self, params: dict, session_id: Optional[str]) -> dict:
		# Single-tab posture: navigate the existing page if a url was given,
		# rather than opening a real new target. Multi-tab is a later milestone.
		url = params.get('url') or 'about:blank'
		if url and url != 'about:blank':
			try:
				await self._adapter().goto(url)
			except Exception:
				pass
		return {'targetId': _SYNTHETIC_TARGET_ID}

	async def _Target_attachToTarget(self, params: dict, session_id: Optional[str]) -> dict:
		# Emit the attachedToTarget event some startup paths wait on, then
		# return the synthetic session id (flatten=True style).
		await self.emit_event(
			'Target.attachedToTarget',
			{
				'sessionId': _SYNTHETIC_SESSION_ID,
				'targetInfo': self._target_info(self._safe_url()),
				'waitingForDebugger': False,
			},
		)
		return {'sessionId': _SYNTHETIC_SESSION_ID}

	async def _Target_closeTarget(self, params: dict, session_id: Optional[str]) -> dict:
		return {'success': True}

	async def _Target_activateTarget(self, params: dict, session_id: Optional[str]) -> dict:
		try:
			await self._page.bring_to_front()
		except Exception:
			pass
		return {}

	# ── Page ────────────────────────────────────────────────────────────
	async def _Page_navigate(self, params: dict, session_id: Optional[str]) -> dict:
		res = await self._adapter().goto(params['url'])
		self._invalidate_scan()
		return {'frameId': _SYNTHETIC_FRAME_ID, 'loaderId': 'BIDI-LOADER-0', 'errorText': None if res else 'navigation failed'}

	async def _Page_reload(self, params: dict, session_id: Optional[str]) -> dict:
		await self._adapter().reload()
		self._invalidate_scan()
		return {}

	async def _Page_getFrameTree(self, params: dict, session_id: Optional[str]) -> dict:
		return {
			'frameTree': {
				'frame': {
					'id': _SYNTHETIC_FRAME_ID,
					'loaderId': 'BIDI-LOADER-0',
					'url': self._safe_url(),
					'securityOrigin': '',
					'mimeType': 'text/html',
				},
				'childFrames': [],
			}
		}

	async def _Page_getNavigationHistory(self, params: dict, session_id: Optional[str]) -> dict:
		return {
			'currentIndex': 0,
			'entries': [{'id': 0, 'url': self._safe_url(), 'userTypedURL': self._safe_url(), 'title': '', 'transitionType': 'typed'}],
		}

	async def _Page_captureScreenshot(self, params: dict, session_id: Optional[str]) -> dict:
		import base64

		full = (params.get('captureBeyondViewport') is True) or False
		png = await self._adapter().screenshot(full_page=full)
		data = png if isinstance(png, (bytes, bytearray)) else b''
		return {'data': base64.b64encode(bytes(data)).decode('ascii')}

	async def _Page_getLayoutMetrics(self, params: dict, session_id: Optional[str]) -> dict:
		m = await self._adapter().viewport_metrics()
		w = int(m.get('width', 0))
		h = int(m.get('height', 0))
		sx = int(m.get('scrollX', 0))
		sy = int(m.get('scrollY', 0))
		cw = int(m.get('contentWidth', w))
		ch = int(m.get('contentHeight', h))
		visual = {'offsetX': 0, 'offsetY': 0, 'pageX': sx, 'pageY': sy, 'clientWidth': w, 'clientHeight': h, 'scale': 1}
		return {
			'layoutViewport': {'pageX': sx, 'pageY': sy, 'clientWidth': w, 'clientHeight': h},
			'visualViewport': visual,
			'contentSize': {'x': 0, 'y': 0, 'width': cw, 'height': ch},
			'cssLayoutViewport': {'pageX': sx, 'pageY': sy, 'clientWidth': w, 'clientHeight': h},
			'cssVisualViewport': visual,
			'cssContentSize': {'x': 0, 'y': 0, 'width': cw, 'height': ch},
		}

	# ── Runtime ─────────────────────────────────────────────────────────
	async def _Runtime_evaluate(self, params: dict, session_id: Optional[str]) -> dict:
		expr = params.get('expression', '')
		by_value = params.get('returnByValue', True)
		try:
			if by_value is False:
				# Caller wants an object reference (objectId), not a value —
				# e.g. dom/service's JS-listener detection. Get a JSHandle.
				handle = await self._page.evaluate_handle(expr)
				is_nullish = await handle.evaluate('x => x === null || x === undefined')
				if is_nullish:
					try:
						await handle.dispose()
					except Exception:
						pass
					return {'result': {'type': 'undefined'}}
				oid = self._nodes.mint_object(handle)
				subtype = 'array' if await handle.evaluate('x => Array.isArray(x)') else None
				res: dict[str, Any] = {'type': 'object', 'objectId': oid}
				if subtype:
					res['subtype'] = subtype
				return {'result': res}
			value = await self._page.evaluate(expr)
			return {'result': _remote_object(value)}
		except Exception as e:  # surface as a CDP exceptionDetails-ish payload
			return {'result': {'type': 'undefined'}, 'exceptionDetails': {'text': str(e)}}

	async def _Runtime_getProperties(self, params: dict, session_id: Optional[str]) -> dict:
		oid = params.get('objectId')
		handle = self._nodes.handle_for_object(oid) if oid else None
		if handle is None:
			return {'result': []}
		props = await handle.get_properties()  # dict[str, JSHandle]
		out = []
		for name, child in props.items():
			child_oid = self._nodes.mint_object(child)
			out.append({'name': name, 'enumerable': True, 'configurable': True,
				'value': {'type': 'object', 'objectId': child_oid}})
		return {'result': out}

	async def _Runtime_releaseObject(self, params: dict, session_id: Optional[str]) -> dict:
		oid = params.get('objectId')
		handle = self._nodes.handle_for_object(oid) if oid else None
		if handle is not None:
			try:
				await handle.dispose()
			except Exception:
				pass
			self._nodes._by_object.pop(oid, None)
		return {}

	# ── DOM / DOMSnapshot (OBSERVE — the synthetic node-identity core) ───
	async def _dom_scan(self) -> dict:
		"""Single source of truth for the DOM tree + flattened snapshot.

		One JS pass walks the document, assigns each element a stable
		``__buid`` (persisted on ``window.__buNext`` so a backendNodeId is
		identical across DOM.getDocument / DOMSnapshot.captureSnapshot /
		DOM.describeNode), and returns BOTH the CDP node tree and the
		flattened snapshot. Cached per page URL; re-scanned on navigation.
		"""
		from browser_use.dom.enhanced_snapshot import REQUIRED_COMPUTED_STYLES

		url = self._safe_url()
		if self._scan_cache is not None and self._scan_url == url:
			return self._scan_cache
		scan = await self._page.evaluate(_DOM_SCAN_JS, REQUIRED_COMPUTED_STYLES)
		self._scan_cache = scan
		self._scan_url = url
		return scan

	def _invalidate_scan(self) -> None:
		self._scan_cache = None
		self._scan_url = None

	async def _DOM_getDocument(self, params: dict, session_id: Optional[str]) -> dict:
		scan = await self._dom_scan()
		return {'root': scan['tree']}

	async def _DOMSnapshot_captureSnapshot(self, params: dict, session_id: Optional[str]) -> dict:
		scan = await self._dom_scan()
		return scan['snapshot']

	async def _DOM_describeNode(self, params: dict, session_id: Optional[str]) -> dict:
		oid = params.get('objectId')
		backend = params.get('backendNodeId')
		buid = None
		if oid is not None:
			handle = self._nodes.handle_for_object(oid)
			if handle is not None:
				try:
					buid = await handle.evaluate(
						'el => (el && el.nodeType === 1) ? (el.__buid || (el.__buid = (window.__buNext = (window.__buNext||0)+1))) : null'
					)
				except Exception:
					buid = None
		elif backend is not None:
			buid = backend
		return {'node': {'backendNodeId': buid, 'nodeId': buid}}

	# ── Accessibility ───────────────────────────────────────────────────
	async def _Accessibility_getFullAXTree(self, params: dict, session_id: Optional[str]) -> dict:
		# v1: no AX merge — elements are still discovered via DOM + snapshot.
		# AX enrichment (role/name keyed by backendNodeId) is a later refinement.
		return {'nodes': []}

	# ── Input ───────────────────────────────────────────────────────────
	async def _Input_dispatchMouseEvent(self, params: dict, session_id: Optional[str]) -> dict:
		x = float(params.get('x', 0))
		y = float(params.get('y', 0))
		etype = params.get('type')
		button = params.get('button', 'left')
		clicks = int(params.get('clickCount', 1) or 1)
		mouse = self._page.mouse
		if etype == 'mouseMoved':
			await mouse.move(x, y)
		elif etype == 'mousePressed':
			await mouse.move(x, y)
			await mouse.down(button=button, click_count=clicks)
		elif etype == 'mouseReleased':
			await mouse.up(button=button, click_count=clicks)
		elif etype == 'mouseWheel':
			await mouse.wheel(float(params.get('deltaX', 0)), float(params.get('deltaY', 0)))
		return {}

	async def _Input_dispatchKeyEvent(self, params: dict, session_id: Optional[str]) -> dict:
		etype = params.get('type')
		kb = self._page.keyboard
		key = params.get('key') or params.get('text') or ''
		if etype == 'keyDown' or etype == 'rawKeyDown':
			if key:
				await kb.down(key)
		elif etype == 'keyUp':
			if key:
				await kb.up(key)
		elif etype == 'char':
			text = params.get('text', '')
			if text:
				await kb.insert_text(text)
		return {}

	async def _Input_insertText(self, params: dict, session_id: Optional[str]) -> dict:
		await self._page.keyboard.insert_text(params.get('text', ''))
		return {}

	# ── helpers ─────────────────────────────────────────────────────────
	def _safe_url(self) -> str:
		try:
			return self._page.url
		except Exception:
			return 'about:blank'

	# ── dispatch table ──────────────────────────────────────────────────
	# method string -> handler. NOTE: enable/disable + interception domains
	# are no-ops on BiDi; DOM-identity domain intentionally absent (raises
	# CdpMethodNotImplemented) until the OBSERVE milestone lands.
	_HANDLERS: dict[str, Callable[['BidiCdpProxy', dict, Optional[str]], Awaitable[dict]]] = {
		# enable/disable no-ops
		'Page.enable': _noop,
		'Page.disable': _noop,
		'DOM.enable': _noop,
		'DOM.disable': _noop,
		'Runtime.enable': _noop,
		'Runtime.disable': _noop,
		'Network.enable': _noop,
		'Network.disable': _noop,
		'Target.setAutoAttach': _noop,
		'Target.setDiscoverTargets': _noop,
		'Page.setLifecycleEventsEnabled': _noop,
		'Runtime.runIfWaitingForDebugger': _noop,
		# Browser
		'Browser.getVersion': _Browser_getVersion,
		# Target
		'Target.getTargets': _Target_getTargets,
		'Target.createTarget': _Target_createTarget,
		'Target.attachToTarget': _Target_attachToTarget,
		'Target.closeTarget': _Target_closeTarget,
		'Target.activateTarget': _Target_activateTarget,
		# Page
		'Page.navigate': _Page_navigate,
		'Page.reload': _Page_reload,
		'Page.getFrameTree': _Page_getFrameTree,
		'Page.getNavigationHistory': _Page_getNavigationHistory,
		'Page.captureScreenshot': _Page_captureScreenshot,
		'Page.getLayoutMetrics': _Page_getLayoutMetrics,
		# Runtime
		'Runtime.evaluate': _Runtime_evaluate,
		'Runtime.getProperties': _Runtime_getProperties,
		'Runtime.releaseObject': _Runtime_releaseObject,
		# DOM / DOMSnapshot / Accessibility (OBSERVE)
		'DOM.getDocument': _DOM_getDocument,
		'DOM.describeNode': _DOM_describeNode,
		'DOMSnapshot.captureSnapshot': _DOMSnapshot_captureSnapshot,
		'Accessibility.getFullAXTree': _Accessibility_getFullAXTree,
		# Input
		'Input.dispatchMouseEvent': _Input_dispatchMouseEvent,
		'Input.dispatchKeyEvent': _Input_dispatchKeyEvent,
		'Input.insertText': _Input_insertText,
	}


def _remote_object(value: Any) -> dict[str, Any]:
	"""Wrap a Python value as a CDP ``Runtime.RemoteObject`` (by-value).
	Object handles (objectId) are minted by the NodeRegistry path, not here."""
	if value is None:
		return {'type': 'undefined'}
	if isinstance(value, bool):
		return {'type': 'boolean', 'value': value}
	if isinstance(value, (int, float)):
		return {'type': 'number', 'value': value}
	if isinstance(value, str):
		return {'type': 'string', 'value': value}
	return {'type': 'object', 'value': value}


# Single-pass DOM walk → CDP node tree + flattened DOMSnapshot. Receives
# REQUIRED_COMPUTED_STYLES as arg[0]. Assigns each element a stable __buid
# (idempotent, persisted on window.__buNext) used as backendNodeId == nodeId
# so the tree, the snapshot, and DOM.describeNode all agree on node identity.
# Bounds are emitted in DEVICE pixels (× devicePixelRatio) because the consumer
# (enhanced_snapshot.build_snapshot_lookup) divides by the device pixel ratio.
_DOM_SCAN_JS = r"""
(requiredStyles) => {
  const dpr = window.devicePixelRatio || 1;
  const strings = [];
  const strMap = new Map();
  const intern = (s) => {
    s = (s == null) ? '' : String(s);
    let i = strMap.get(s);
    if (i === undefined) { i = strings.length; strings.push(s); strMap.set(s, i); }
    return i;
  };
  if (typeof window.__buNext !== 'number') window.__buNext = 0;
  const nextId = () => (window.__buNext = window.__buNext + 1);

  const backendNodeId = [];
  const nodeIndex = [], bounds = [], styles = [], paintOrders = [],
        clientRects = [], scrollRects = [];
  let snapIdx = 0;
  let paint = 0;

  function walk(node) {
    const isEl = node.nodeType === 1;
    // Stable id: reuse an element's existing __buid; mint otherwise.
    const id = isEl ? (node.__buid || (node.__buid = nextId())) : nextId();
    const si = snapIdx++;
    backendNodeId.push(id);

    if (isEl) {
      let r;
      try { r = node.getBoundingClientRect(); } catch (e) { r = {x:0,y:0,width:0,height:0}; }
      nodeIndex.push(si);
      bounds.push([r.x * dpr, r.y * dpr, r.width * dpr, r.height * dpr]);
      let cs; try { cs = getComputedStyle(node); } catch (e) { cs = null; }
      styles.push(requiredStyles.map(p => intern(cs ? cs.getPropertyValue(p) : '')));
      paintOrders.push(paint++);
      clientRects.push([r.x * dpr, r.y * dpr, r.width * dpr, r.height * dpr]);
      scrollRects.push([(node.scrollLeft||0)*dpr, (node.scrollTop||0)*dpr,
                        (node.scrollWidth||0)*dpr, (node.scrollHeight||0)*dpr]);
    }

    const treeNode = {
      nodeId: id,
      backendNodeId: id,
      nodeType: node.nodeType,
      nodeName: node.nodeName,
      nodeValue: node.nodeValue || '',
    };
    if (isEl) {
      const attrs = [];
      for (const a of node.attributes) attrs.push(a.name, a.value);
      treeNode.attributes = attrs;
    }
    const kids = [];
    const childNodes = node.childNodes || [];
    for (const c of childNodes) {
      // Drop pure-whitespace text nodes to keep the tree lean.
      if (c.nodeType === 3 && !(c.nodeValue && c.nodeValue.trim())) continue;
      // Skip non-rendered nodes that bloat the tree.
      if (c.nodeType === 8) continue; // comments
      kids.push(walk(c));
    }
    if (kids.length) { treeNode.children = kids; treeNode.childNodeCount = kids.length; }
    return treeNode;
  }

  const tree = walk(document);
  const snapshot = {
    documents: [{
      documentURL: intern(document.URL),
      title: intern(document.title || ''),
      nodes: { backendNodeId: backendNodeId },
      layout: {
        nodeIndex: nodeIndex,
        bounds: bounds,
        styles: styles,
        paintOrders: paintOrders,
        clientRects: clientRects,
        scrollRects: scrollRects,
        // NB: do NOT emit stackingContexts — the consumer guards it with
        // len(dict)==1 then indexes index[layout_idx], which IndexErrors on
        // an empty/short index list. Omitting it makes the guard (len([])==0)
        // skip cleanly. Stacking-context z-order is not needed for v1.
      },
    }],
    strings: strings,
  };
  return { tree: tree, snapshot: snapshot };
}
"""


__all__ = ['BidiCdpProxy', 'NodeRegistry', 'CdpMethodNotImplemented']
