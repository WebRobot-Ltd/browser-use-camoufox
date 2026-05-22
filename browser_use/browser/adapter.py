"""Browser adapter — the page-level convergence point.

The library historically drives Chromium via raw CDP (the cdp_use library).
To support Firefox / Camoufox we need Playwright Firefox or BiDi-based
drivers. Those protocol stacks are incompatible, so the consumers of
"a browser page" — DOM service, action watchdogs, the agentic loop —
must talk through a common contract and stay protocol-agnostic.

That contract is :class:`BrowserAdapter`. Two concrete implementations
ship with this module:

  - :class:`CdpBrowserAdapter` synthesises every operation from raw CDP
    calls against an existing :class:`~browser_use.browser.session.BrowserSession`.
    Most operations are wrapped JS evaluations (``Runtime.evaluate``),
    a couple need real CDP commands (``Page.captureScreenshot``,
    ``Page.navigate``). It's the bridge from the existing codebase to
    the adapter layer — no Playwright dependency.

  - :class:`PlaywrightBrowserAdapter` wraps a :class:`playwright.async_api.Page`.
    Most methods are one-line passthroughs because the adapter contract
    is intentionally modelled on Playwright's Page API. This is the
    path for Firefox / Camoufox.

See ``docs/ADAPTERS.md`` for the conventions when writing a third backend.

Note on scope: this module is **operations only** — how you control a
page, not how you launched it. The launch path lives in
:mod:`browser_use.browser.engine`. The two layers are orthogonal.
"""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
	from browser_use.browser.session import BrowserSession

	try:
		from playwright.async_api import Page as PlaywrightPage
	except ImportError:  # pragma: no cover — playwright is optional
		PlaywrightPage = Any  # type: ignore


# ── ABC ──────────────────────────────────────────────────────────────────────


class BrowserAdapter(ABC):
	"""The page-level contract every backend implements.

	The method set is intentionally a strict subset of Playwright's async
	Page API. Adding a new operation means adding it here (with an
	abstract signature) and in every concrete adapter — never call into a
	backend-specific API from a consumer. If you find yourself wanting
	to, the operation belongs on this contract.

	State semantics:
	  - All methods are ``async``.
	  - Timeouts are in **milliseconds**, matching Playwright (and the
	    most common operations on the CDP side).
	  - Selectors are **CSS selectors** (no XPath, no Playwright-specific
	    role= / text= selectors at this layer — keep the adapter
	    backend-portable). Build higher-level locator abstractions on
	    top.
	  - Methods that operate on an element raise on "no element matched"
	    rather than returning ``None``, so consumers can use
	    ``locator_count() > 0`` to test existence first.
	"""

	# ── Navigation ──

	@abstractmethod
	async def goto(self, url: str, *, wait_until: str = 'load',
	               timeout_ms: int = 30_000) -> dict[str, Any]:
		"""Navigate to ``url``. ``wait_until`` mirrors Playwright's
		``"load" | "domcontentloaded" | "networkidle" | "commit"``.

		Returns ``{"url": str, "status": int | None}`` — the final URL
		after any redirects, plus the HTTP status of the main resource
		when the backend can surface it (Playwright can; raw CDP-eval
		cannot, in which case ``status`` is ``None``).
		"""

	@abstractmethod
	async def reload(self, *, wait_until: str = 'load',
	                 timeout_ms: int = 30_000) -> None: ...

	@abstractmethod
	async def url(self) -> str:
		"""Current page URL (after any client-side navigation)."""

	@abstractmethod
	async def title(self) -> str:
		"""Current ``document.title``."""

	@abstractmethod
	async def wait_for_load_state(self, state: str = 'load',
	                              *, timeout_ms: int = 30_000) -> None: ...

	# ── Content + script ──

	@abstractmethod
	async def content(self) -> str:
		"""Return the current ``document.documentElement.outerHTML``."""

	@abstractmethod
	async def evaluate(self, expression: str, arg: Any = None) -> Any:
		"""Run JavaScript in the page context. Returns whatever the
		expression evaluates to (serialised through JSON). ``arg`` is
		passed as the function argument when ``expression`` is a function
		body (Playwright convention).
		"""

	# ── Locator-style queries ──

	@abstractmethod
	async def locator_count(self, css_selector: str) -> int:
		"""Number of elements matching ``css_selector``. 0 if none."""

	@abstractmethod
	async def locator_inner_text(self, css_selector: str, *,
	                             timeout_ms: int = 2_000) -> str:
		"""``innerText`` of the first matching element. Raises if zero
		matches before ``timeout_ms``."""

	@abstractmethod
	async def locator_get_attribute(self, css_selector: str,
	                                attribute: str, *,
	                                timeout_ms: int = 2_000) -> str | None:
		"""Attribute value of the first matching element, or ``None`` if
		the attribute is absent. Raises if zero matches."""

	@abstractmethod
	async def locator_is_visible(self, css_selector: str, *,
	                             timeout_ms: int = 2_000) -> bool:
		"""``True`` if the first matching element is visually rendered
		(non-zero box, not ``display: none`` / ``visibility: hidden``).
		``False`` if zero matches or hidden."""

	# ── Element actions ──

	@abstractmethod
	async def click(self, css_selector: str, *,
	                timeout_ms: int = 5_000) -> None: ...

	@abstractmethod
	async def fill(self, css_selector: str, value: str, *,
	               timeout_ms: int = 5_000) -> None:
		"""Clear the input and type ``value`` into it (Playwright's
		``fill`` semantics — atomic, no per-keystroke listeners fire)."""

	@abstractmethod
	async def press(self, css_selector: str, key: str, *,
	                timeout_ms: int = 5_000) -> None:
		"""Press a key on the focused element (Playwright key names:
		``"Enter"``, ``"Escape"``, ``"ArrowDown"``, …)."""

	@abstractmethod
	async def hover(self, css_selector: str, *,
	                timeout_ms: int = 5_000) -> None: ...

	# ── Waiting ──

	@abstractmethod
	async def wait_for_selector(self, css_selector: str, *,
	                            state: str = 'visible',
	                            timeout_ms: int = 30_000) -> None:
		"""Block until at least one element matches and is in ``state``
		(``"attached" | "detached" | "visible" | "hidden"``)."""

	# ── Page state ──

	@abstractmethod
	async def set_viewport_size(self, width: int, height: int) -> None: ...

	# ── Viewport / metrics ──

	@abstractmethod
	async def device_pixel_ratio(self) -> float:
		"""``window.devicePixelRatio`` — the CSS-px-to-device-px ratio. 1.0
		on standard displays, 2.0+ on HiDPI / Retina. Needed by any
		consumer that converts between page coordinates and screen
		coordinates (DOM service viewport math, screenshot overlays,
		click coordinate dispatch)."""

	@abstractmethod
	async def viewport_metrics(self) -> dict[str, Any]:
		"""Layout-level dimensions of the current viewport. Returns a dict
		with at least::

		    {
		      'width':       int,    # CSS px — visual viewport width
		      'height':      int,    # CSS px — visual viewport height
		      'scroll_x':    int,    # CSS px — horizontal scroll offset
		      'scroll_y':    int,    # CSS px — vertical scroll offset
		      'document_width':  int,
		      'document_height': int,
		      'device_pixel_ratio': float,
		    }

		Backends MAY include additional keys (CDP's getLayoutMetrics
		surfaces 6+ tiers, Playwright exposes fewer). Consumers should
		only depend on the keys above and treat the rest as best-effort."""

	# ── Media ──

	@abstractmethod
	async def screenshot(self, *, full_page: bool = False,
	                     fmt: Literal['png', 'jpeg'] = 'png') -> bytes: ...

	# ── Accessibility ──

	@abstractmethod
	async def accessibility_snapshot(self, *,
	                                 interesting_only: bool = True) -> dict | None:
		"""Cross-browser accessibility tree snapshot. The CDP backend
		synthesises it from ``Accessibility.getFullAXTree``; Playwright
		exposes it directly. Returns ``None`` if the backend can't
		produce one."""

	# ── Cookies ──

	@abstractmethod
	async def get_cookies(self, urls: list[str] | None = None) -> list[dict[str, Any]]: ...

	@abstractmethod
	async def set_cookies(self, cookies: list[dict[str, Any]]) -> None: ...

	# ── Lifecycle ──

	@abstractmethod
	async def close(self) -> None:
		"""Release whatever the adapter owns (typically nothing — the
		owner of the underlying session/page handles teardown). Always
		safe to call multiple times."""


# ── CDP-backed adapter ───────────────────────────────────────────────────────


class CdpBrowserAdapter(BrowserAdapter):
	"""Page-level operations synthesised against a CDP session.

	Wraps an existing :class:`BrowserSession` and dispatches every
	operation through ``session.cdp_client.send.<Domain>.<method>(...)``.
	Most operations are JS evaluations via ``Runtime.evaluate`` — the
	simplest cross-backend implementation. Where CDP has a first-class
	command (``Page.navigate``, ``Page.captureScreenshot``,
	``Network.getCookies``) we use it directly.

	This adapter is the bridge from the legacy CDP-everywhere codebase
	to the adapter layer. Pre-existing code paths in
	:mod:`browser_use.browser.session` are not affected — consumers
	choose to use the adapter incrementally.
	"""

	def __init__(self, session: BrowserSession,
	             cdp_session_id: str | None = None) -> None:
		"""Wrap a BrowserSession. When ``cdp_session_id`` is provided every
		CDP call routes to that specific target (tab/iframe). When
		``None``, calls route to ``session.current_session_id`` — the
		"current tab" — which is the default for the agentic loop.

		Use :meth:`for_target` to build a target-pinned instance from a
		target_id (the way :class:`DomService` works with per-frame
		sessions)."""
		self._session = session
		self._cdp_session_id = cdp_session_id

	@classmethod
	async def for_target(cls, session: BrowserSession, target_id: str,
	                     *, focus: bool = False) -> CdpBrowserAdapter:
		"""Build an adapter pinned to a specific CDP target (tab/iframe).

		Used by consumers that walk multiple frames (DomService,
		cross-frame action handlers). For "just talk to the current tab"
		use the bare constructor."""
		cdp_session = await session.get_or_create_cdp_session(
			target_id=target_id, focus=focus,
		)
		return cls(session, cdp_session_id=cdp_session.session_id)

	# Helpers ----------------------------------------------------------------

	def _session_id(self) -> str:
		"""Resolve the CDP session_id to route a command to. Pinned target
		wins when set; otherwise tracks ``current_session_id``."""
		return self._cdp_session_id or self._session.current_session_id

	async def _eval(self, expression: str, *, return_by_value: bool = True) -> Any:
		"""``Runtime.evaluate`` round-trip. Returns the unwrapped value."""
		cdp = self._session.cdp_client
		result = await cdp.send.Runtime.evaluate(
			params={'expression': expression, 'returnByValue': return_by_value,
			        'awaitPromise': True},
			session_id=self._session_id(),
		)
		if result.get('exceptionDetails'):
			detail = result['exceptionDetails']
			raise RuntimeError(f'page eval threw: {detail.get("text", detail)}')
		return result.get('result', {}).get('value')

	@staticmethod
	def _css_literal(css: str) -> str:
		"""Embed a CSS selector safely in a JS string literal."""
		return json.dumps(css)

	# Navigation -------------------------------------------------------------

	async def goto(self, url: str, *, wait_until: str = 'load',
	               timeout_ms: int = 30_000) -> dict[str, Any]:
		cdp = self._session.cdp_client
		await cdp.send.Page.navigate(
			params={'url': url, 'transitionType': 'typed'},
			session_id=self._session_id(),
		)
		await self.wait_for_load_state(wait_until, timeout_ms=timeout_ms)
		return {'url': await self.url(), 'status': None}  # CDP doesn't surface main-frame status here

	async def reload(self, *, wait_until: str = 'load',
	                 timeout_ms: int = 30_000) -> None:
		cdp = self._session.cdp_client
		await cdp.send.Page.reload(params={}, session_id=self._session_id())
		await self.wait_for_load_state(wait_until, timeout_ms=timeout_ms)

	async def url(self) -> str:
		return await self._eval('document.location.href')

	async def title(self) -> str:
		return await self._eval('document.title')

	async def wait_for_load_state(self, state: str = 'load',
	                              *, timeout_ms: int = 30_000) -> None:
		# Best-effort cross-browser implementation via readyState polling.
		# A production-grade impl listens to Page.lifecycleEvent.
		# 'load' → readyState == 'complete'; 'domcontentloaded' → 'interactive' or 'complete'.
		target_states = {
			'load': ('complete',),
			'domcontentloaded': ('interactive', 'complete'),
			'networkidle': ('complete',),  # CDP networkIdle is harder; approximate via load
			'commit': ('loading', 'interactive', 'complete'),
		}.get(state, ('complete',))
		# Poll up to timeout_ms.
		import asyncio
		deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000)
		while asyncio.get_event_loop().time() < deadline:
			rs = await self._eval('document.readyState')
			if rs in target_states:
				return
			await asyncio.sleep(0.05)
		raise TimeoutError(f'wait_for_load_state({state!r}) timed out after {timeout_ms}ms')

	# Content + script -------------------------------------------------------

	async def content(self) -> str:
		return await self._eval('document.documentElement.outerHTML')

	async def evaluate(self, expression: str, arg: Any = None) -> Any:
		# Playwright accepts both ``"() => 1+1"`` (arrow-function-as-string)
		# and ``"1+1"`` (bare expression). CDP's Runtime.evaluate only
		# handles bare expressions. Detect arrow function and lift it.
		if expression.lstrip().startswith(('(', 'function')):
			# Wrap as IIFE so we can supply ``arg``.
			arg_literal = json.dumps(arg) if arg is not None else 'undefined'
			expression = f'({expression})({arg_literal})'
		return await self._eval(expression)

	# Locator-style queries --------------------------------------------------

	async def locator_count(self, css_selector: str) -> int:
		js = f'document.querySelectorAll({self._css_literal(css_selector)}).length'
		return await self._eval(js)

	async def _wait_for_match(self, css_selector: str, timeout_ms: int) -> None:
		"""Block until ``locator_count > 0``; raise on timeout."""
		import asyncio
		deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000)
		while asyncio.get_event_loop().time() < deadline:
			if await self.locator_count(css_selector) > 0:
				return
			await asyncio.sleep(0.05)
		raise TimeoutError(f'no element matched {css_selector!r} within {timeout_ms}ms')

	async def locator_inner_text(self, css_selector: str, *,
	                             timeout_ms: int = 2_000) -> str:
		await self._wait_for_match(css_selector, timeout_ms)
		js = (
			f'(() => {{ const el = document.querySelector({self._css_literal(css_selector)}); '
			f'return el ? el.innerText : null; }})()'
		)
		out = await self._eval(js)
		if out is None:
			raise RuntimeError(f'no element matched {css_selector!r}')
		return out

	async def locator_get_attribute(self, css_selector: str, attribute: str, *,
	                                timeout_ms: int = 2_000) -> str | None:
		await self._wait_for_match(css_selector, timeout_ms)
		js = (
			f'(() => {{ const el = document.querySelector({self._css_literal(css_selector)}); '
			f'return el ? el.getAttribute({json.dumps(attribute)}) : null; }})()'
		)
		return await self._eval(js)

	async def locator_is_visible(self, css_selector: str, *,
	                             timeout_ms: int = 2_000) -> bool:
		if await self.locator_count(css_selector) == 0:
			return False
		js = (
			f'(() => {{ const el = document.querySelector({self._css_literal(css_selector)}); '
			f'if (!el) return false; '
			f'const rect = el.getBoundingClientRect(); '
			f'const style = window.getComputedStyle(el); '
			f'return rect.width > 0 && rect.height > 0 '
			f'&& style.visibility !== "hidden" && style.display !== "none"; '
			f'}})()'
		)
		return bool(await self._eval(js))

	# Element actions --------------------------------------------------------
	#
	# We use JS-level dispatch (``element.click()``, ``element.dispatchEvent``)
	# rather than synthesising real mouse/keyboard events via
	# Input.dispatchMouseEvent / Input.dispatchKeyEvent. JS dispatch is more
	# portable across backends and good enough for the selector-grounding
	# use case. For interactions that legitimately need OS-level event
	# fidelity (drag, hardware-only listeners), the consumer should use a
	# backend-specific path until we add an "input" tier to the adapter.

	async def click(self, css_selector: str, *, timeout_ms: int = 5_000) -> None:
		await self._wait_for_match(css_selector, timeout_ms)
		js = (
			f'(() => {{ const el = document.querySelector({self._css_literal(css_selector)}); '
			f'if (!el) throw new Error("no element matched"); '
			f'el.click(); return true; }})()'
		)
		await self._eval(js)

	async def fill(self, css_selector: str, value: str, *,
	               timeout_ms: int = 5_000) -> None:
		await self._wait_for_match(css_selector, timeout_ms)
		js = (
			f'(() => {{ const el = document.querySelector({self._css_literal(css_selector)}); '
			f'if (!el) throw new Error("no element matched"); '
			f'el.focus(); el.value = {json.dumps(value)}; '
			f'el.dispatchEvent(new Event("input", {{bubbles: true}})); '
			f'el.dispatchEvent(new Event("change", {{bubbles: true}})); '
			f'return true; }})()'
		)
		await self._eval(js)

	async def press(self, css_selector: str, key: str, *,
	                timeout_ms: int = 5_000) -> None:
		await self._wait_for_match(css_selector, timeout_ms)
		# Playwright-style key names map directly to KeyboardEvent.key for
		# the common cases (Enter, Escape, Tab, ArrowDown, …). Single-char
		# keys are case-sensitive.
		js = (
			f'(() => {{ const el = document.querySelector({self._css_literal(css_selector)}); '
			f'if (!el) throw new Error("no element matched"); el.focus(); '
			f'for (const t of ["keydown", "keypress", "keyup"]) {{ '
			f'  el.dispatchEvent(new KeyboardEvent(t, {{key: {json.dumps(key)}, bubbles: true}})); '
			f'}} return true; }})()'
		)
		await self._eval(js)

	async def hover(self, css_selector: str, *, timeout_ms: int = 5_000) -> None:
		await self._wait_for_match(css_selector, timeout_ms)
		js = (
			f'(() => {{ const el = document.querySelector({self._css_literal(css_selector)}); '
			f'if (!el) throw new Error("no element matched"); '
			f'const r = el.getBoundingClientRect(); '
			f'el.dispatchEvent(new MouseEvent("mouseover", '
			f'  {{bubbles: true, clientX: r.left + r.width/2, clientY: r.top + r.height/2}})); '
			f'return true; }})()'
		)
		await self._eval(js)

	# Waiting ----------------------------------------------------------------

	async def wait_for_selector(self, css_selector: str, *,
	                            state: str = 'visible',
	                            timeout_ms: int = 30_000) -> None:
		import asyncio
		deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000)
		while asyncio.get_event_loop().time() < deadline:
			count = await self.locator_count(css_selector)
			if state == 'attached'  and count > 0: return
			if state == 'detached'  and count == 0: return
			if state == 'visible'   and count > 0 and await self.locator_is_visible(css_selector, timeout_ms=100): return
			if state == 'hidden'    and (count == 0 or not await self.locator_is_visible(css_selector, timeout_ms=100)): return
			await asyncio.sleep(0.05)
		raise TimeoutError(f'wait_for_selector({css_selector!r}, state={state!r}) timed out')

	# Page state -------------------------------------------------------------

	async def set_viewport_size(self, width: int, height: int) -> None:
		cdp = self._session.cdp_client
		await cdp.send.Emulation.setDeviceMetricsOverride(
			params={'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': False},
			session_id=self._session_id(),
		)

	# Viewport / metrics -----------------------------------------------------

	async def device_pixel_ratio(self) -> float:
		# CDP's Page.getLayoutMetrics returns visualViewport (device px) and
		# cssVisualViewport (CSS px); their ratio is the DPR. We could also
		# `Runtime.evaluate('window.devicePixelRatio')` but going through
		# getLayoutMetrics is more reliable in detached frames where JS may
		# not yet be ready.
		try:
			cdp = self._session.cdp_client
			metrics = await cdp.send.Page.getLayoutMetrics(session_id=self._session_id())
			visual = metrics.get('visualViewport', {})
			css_visual = metrics.get('cssVisualViewport', {})
			device_w = float(visual.get('clientWidth') or 0.0)
			css_w = float(css_visual.get('clientWidth') or 0.0)
			if css_w > 0 and device_w > 0:
				return device_w / css_w
		except Exception:
			pass
		# Fallback via JS — fires only if getLayoutMetrics is unavailable.
		try:
			return float(await self._eval('window.devicePixelRatio'))
		except Exception:
			return 1.0

	async def viewport_metrics(self) -> dict[str, Any]:
		cdp = self._session.cdp_client
		try:
			metrics = await cdp.send.Page.getLayoutMetrics(session_id=self._session_id())
		except Exception:
			# Fallback to JS-only viewport — no CDP available.
			vm = await self._eval(
				'(() => ({w: innerWidth, h: innerHeight, sx: scrollX, sy: scrollY, '
				'dw: document.documentElement.scrollWidth, '
				'dh: document.documentElement.scrollHeight, '
				'dpr: window.devicePixelRatio}))()'
			)
			return {
				'width':              int(vm.get('w', 0)),
				'height':             int(vm.get('h', 0)),
				'scroll_x':           int(vm.get('sx', 0)),
				'scroll_y':           int(vm.get('sy', 0)),
				'document_width':     int(vm.get('dw', 0)),
				'document_height':    int(vm.get('dh', 0)),
				'device_pixel_ratio': float(vm.get('dpr', 1.0)),
			}
		# CDP-only normalised shape. We expose only what we can guarantee
		# the field exists across Chromium versions; backend-specific
		# extras stay in metrics['_raw'] for advanced consumers.
		css_visual = metrics.get('cssVisualViewport', {})
		css_layout = metrics.get('cssLayoutViewport', {})
		content_size = metrics.get('cssContentSize') or metrics.get('contentSize', {})
		visual = metrics.get('visualViewport', {})
		device_w = float(visual.get('clientWidth') or css_visual.get('clientWidth', 0))
		css_w = float(css_visual.get('clientWidth', 0))
		dpr = device_w / css_w if css_w > 0 else 1.0
		return {
			'width':              int(css_visual.get('clientWidth', css_layout.get('clientWidth', 0))),
			'height':             int(css_visual.get('clientHeight', css_layout.get('clientHeight', 0))),
			'scroll_x':           int(css_visual.get('pageX', 0)),
			'scroll_y':           int(css_visual.get('pageY', 0)),
			'document_width':     int(content_size.get('width', 0)),
			'document_height':    int(content_size.get('height', 0)),
			'device_pixel_ratio': dpr,
			'_raw':               metrics,
		}

	# Media ------------------------------------------------------------------

	async def screenshot(self, *, full_page: bool = False,
	                     fmt: Literal['png', 'jpeg'] = 'png') -> bytes:
		cdp = self._session.cdp_client
		result = await cdp.send.Page.captureScreenshot(
			params={'format': fmt, 'captureBeyondViewport': full_page},
			session_id=self._session_id(),
		)
		return base64.b64decode(result['data'])

	# Accessibility ----------------------------------------------------------

	async def accessibility_snapshot(self, *,
	                                 interesting_only: bool = True) -> dict | None:
		cdp = self._session.cdp_client
		try:
			result = await cdp.send.Accessibility.getFullAXTree(
				params={}, session_id=self._session_id(),
			)
		except Exception:
			return None
		nodes = result.get('nodes', [])
		if interesting_only:
			# Filter ignored / decorative nodes. Same posture as Playwright's
			# ``interesting_only=True`` (ignored=False AND name/role present).
			nodes = [n for n in nodes if not n.get('ignored', True) and (n.get('name') or n.get('role'))]
		return {'nodes': nodes}

	# Cookies ----------------------------------------------------------------

	async def get_cookies(self, urls: list[str] | None = None) -> list[dict[str, Any]]:
		cdp = self._session.cdp_client
		params = {'urls': urls} if urls else {}
		result = await cdp.send.Network.getCookies(
			params=params, session_id=self._session_id(),
		)
		return result.get('cookies', [])

	async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
		cdp = self._session.cdp_client
		await cdp.send.Network.setCookies(
			params={'cookies': cookies}, session_id=self._session_id(),
		)

	# Lifecycle --------------------------------------------------------------

	async def close(self) -> None:
		# CdpBrowserAdapter doesn't own the session — the caller does. No-op.
		return None


# ── Playwright-backed adapter ────────────────────────────────────────────────


class PlaywrightBrowserAdapter(BrowserAdapter):
	"""Page-level operations forwarded to a Playwright ``Page``.

	The contract is modelled on Playwright's API, so most methods are
	one-liner passthroughs. This is the entry point for Firefox /
	Camoufox: instantiate a Playwright Firefox page (or hand off a
	Camoufox-launched page), wrap it, and the rest of the agentic loop
	works unchanged.
	"""

	def __init__(self, page: PlaywrightPage) -> None:
		self._page = page

	# Navigation -------------------------------------------------------------

	async def goto(self, url: str, *, wait_until: str = 'load',
	               timeout_ms: int = 30_000) -> dict[str, Any]:
		response = await self._page.goto(url, wait_until=wait_until, timeout=timeout_ms)
		return {'url': self._page.url, 'status': response.status if response else None}

	async def reload(self, *, wait_until: str = 'load',
	                 timeout_ms: int = 30_000) -> None:
		await self._page.reload(wait_until=wait_until, timeout=timeout_ms)

	async def url(self) -> str:
		return self._page.url

	async def title(self) -> str:
		return await self._page.title()

	async def wait_for_load_state(self, state: str = 'load',
	                              *, timeout_ms: int = 30_000) -> None:
		await self._page.wait_for_load_state(state, timeout=timeout_ms)

	# Content + script -------------------------------------------------------

	async def content(self) -> str:
		return await self._page.content()

	async def evaluate(self, expression: str, arg: Any = None) -> Any:
		return await self._page.evaluate(expression, arg)

	# Locator-style queries --------------------------------------------------

	async def locator_count(self, css_selector: str) -> int:
		return await self._page.locator(css_selector).count()

	async def locator_inner_text(self, css_selector: str, *,
	                             timeout_ms: int = 2_000) -> str:
		return await self._page.locator(css_selector).first.inner_text(timeout=timeout_ms)

	async def locator_get_attribute(self, css_selector: str, attribute: str, *,
	                                timeout_ms: int = 2_000) -> str | None:
		return await self._page.locator(css_selector).first.get_attribute(attribute, timeout=timeout_ms)

	async def locator_is_visible(self, css_selector: str, *,
	                             timeout_ms: int = 2_000) -> bool:
		try:
			return await self._page.locator(css_selector).first.is_visible(timeout=timeout_ms)
		except Exception:
			return False

	# Element actions --------------------------------------------------------

	async def click(self, css_selector: str, *, timeout_ms: int = 5_000) -> None:
		await self._page.locator(css_selector).first.click(timeout=timeout_ms)

	async def fill(self, css_selector: str, value: str, *,
	               timeout_ms: int = 5_000) -> None:
		await self._page.locator(css_selector).first.fill(value, timeout=timeout_ms)

	async def press(self, css_selector: str, key: str, *,
	                timeout_ms: int = 5_000) -> None:
		await self._page.locator(css_selector).first.press(key, timeout=timeout_ms)

	async def hover(self, css_selector: str, *, timeout_ms: int = 5_000) -> None:
		await self._page.locator(css_selector).first.hover(timeout=timeout_ms)

	# Waiting ----------------------------------------------------------------

	async def wait_for_selector(self, css_selector: str, *,
	                            state: str = 'visible',
	                            timeout_ms: int = 30_000) -> None:
		await self._page.wait_for_selector(css_selector, state=state, timeout=timeout_ms)

	# Page state -------------------------------------------------------------

	async def set_viewport_size(self, width: int, height: int) -> None:
		await self._page.set_viewport_size({'width': width, 'height': height})

	# Viewport / metrics -----------------------------------------------------

	async def device_pixel_ratio(self) -> float:
		return float(await self._page.evaluate('() => window.devicePixelRatio'))

	async def viewport_metrics(self) -> dict[str, Any]:
		vm = await self._page.evaluate(
			'() => ({w: innerWidth, h: innerHeight, sx: scrollX, sy: scrollY, '
			'dw: document.documentElement.scrollWidth, '
			'dh: document.documentElement.scrollHeight, '
			'dpr: window.devicePixelRatio})'
		)
		return {
			'width':              int(vm['w']),
			'height':             int(vm['h']),
			'scroll_x':           int(vm['sx']),
			'scroll_y':           int(vm['sy']),
			'document_width':     int(vm['dw']),
			'document_height':    int(vm['dh']),
			'device_pixel_ratio': float(vm['dpr']),
		}

	# Media ------------------------------------------------------------------

	async def screenshot(self, *, full_page: bool = False,
	                     fmt: Literal['png', 'jpeg'] = 'png') -> bytes:
		return await self._page.screenshot(full_page=full_page, type=fmt)

	# Accessibility ----------------------------------------------------------

	async def accessibility_snapshot(self, *,
	                                 interesting_only: bool = True) -> dict | None:
		try:
			return await self._page.accessibility.snapshot(interesting_only=interesting_only)
		except Exception:
			return None

	# Cookies ----------------------------------------------------------------

	async def get_cookies(self, urls: list[str] | None = None) -> list[dict[str, Any]]:
		return await self._page.context.cookies(urls or [])

	async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
		await self._page.context.add_cookies(cookies)

	# Lifecycle --------------------------------------------------------------

	async def close(self) -> None:
		# The adapter doesn't own the page either — the caller manages
		# the Playwright lifecycle (so visit/observe/visit again from the
		# same actor reuses the same page).
		return None


__all__ = [
	'BrowserAdapter',
	'CdpBrowserAdapter',
	'PlaywrightBrowserAdapter',
]
