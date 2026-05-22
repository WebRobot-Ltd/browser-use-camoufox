"""Browser connection — the bring-up + lifetime layer.

Phase 5a of the firefox-compat porting effort. Where :mod:`adapter`
abstracts *page-level operations* (goto, click, evaluate), this module
abstracts the *session-level resources* a :class:`BrowserSession` owns:

  - The long-lived control channel to the browser process (a CDP
    WebSocket for Chromium, a Playwright BiDi connection for Firefox).
  - Lifecycle (``start`` / ``stop`` of that channel).
  - Target discovery — listing tabs/pages, attaching to a specific one.

A :class:`BrowserConnection` is intentionally *narrower* than a full
Playwright Browser handle: it surfaces just enough for
:class:`BrowserSession` to bring itself up without leaking
protocol-specific types into the rest of browser-use.

Two implementations ship here:

  - :class:`CdpBrowserConnection` wraps the existing ``cdp_use.CDPClient``.
    Mirrors what :meth:`BrowserSession.connect` already does today —
    nothing functionally new on the Chromium path.

  - :class:`BidiBrowserConnection` wraps a Playwright ``Browser``
    obtained by connecting to a Firefox / Camoufox WebSocket endpoint
    (or by launching one in-process via the engine). Provides the same
    surface so :class:`BrowserSession` can bring up cleanly on Firefox
    targets.

What this commit does NOT do:

  - Refactor every watchdog to consume the connection (Phase 5b/c/d).
  - Make ``Agent.run`` work end-to-end on Firefox — most watchdogs
    still talk raw ``session.cdp_client.send.*`` and would fail on a
    BidiBrowserConnection. The path opens; the trail still needs
    cutting. See ``docs/ADAPTERS.md`` for the boundary doctrine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
	from cdp_use import CDPClient

	try:
		from playwright.async_api import Browser as PlaywrightBrowser
		from playwright.async_api import BrowserContext as PlaywrightContext
		from playwright.async_api import Page as PlaywrightPage
		from playwright.async_api import Playwright
	except ImportError:  # pragma: no cover — playwright is optional on CDP-only deploys
		PlaywrightBrowser = Any  # type: ignore
		PlaywrightContext = Any  # type: ignore
		PlaywrightPage = Any  # type: ignore
		Playwright = Any  # type: ignore


class BrowserConnection(ABC):
	"""The contract every browser-control connection implements.

	An instance owns the network channel + lifecycle of one logical
	browser. Long-lived: created on session start, torn down on session
	stop.

	The :attr:`backend` attribute returns a stable short identifier
	(``'cdp'`` or ``'bidi'``) so caller code can branch when it has to
	(legacy paths that haven't been refactored to the
	:class:`~browser_use.browser.adapter.BrowserAdapter` yet).
	"""

	#: Stable identifier — ``'cdp'`` or ``'bidi'``.
	backend: str = 'base'

	@abstractmethod
	async def start(self) -> None:
		"""Open the underlying channel. Idempotent: safe to call twice."""

	@abstractmethod
	async def stop(self) -> None:
		"""Tear down the channel. Idempotent: safe to call after start
		failed or after a previous stop."""

	@property
	@abstractmethod
	def is_open(self) -> bool:
		"""``True`` between a successful :meth:`start` and the first
		:meth:`stop` (or an underlying connection failure)."""


# ── CDP backend ──────────────────────────────────────────────────────────────


class CdpBrowserConnection(BrowserConnection):
	"""Wraps :class:`cdp_use.CDPClient`. Used by every Chromium target.

	The connection is constructed *around* an existing :class:`CDPClient`
	instance — that matches how :class:`BrowserSession.connect` currently
	builds the client (with retry, timeout, additional_headers), and lets
	this class stay agnostic about *how* the client was built.
	"""

	backend = 'cdp'

	def __init__(self, cdp_client: CDPClient) -> None:
		self._client = cdp_client
		self._started = False

	@property
	def client(self) -> CDPClient:
		"""Access the underlying :class:`CDPClient` — used by legacy
		paths in :mod:`browser_use.browser.session` and the watchdogs
		that haven't been refactored to the
		:class:`~browser_use.browser.adapter.BrowserAdapter` yet."""
		return self._client

	async def start(self) -> None:
		if self._started:
			return
		await self._client.start()
		self._started = True

	async def stop(self) -> None:
		if not self._started:
			return
		try:
			await self._client.stop()
		finally:
			self._started = False

	@property
	def is_open(self) -> bool:
		if not self._started:
			return False
		ws = getattr(self._client, 'ws', None)
		if ws is None:
			return False
		# cdp_use uses an internal State enum on its WS wrapper. Coerce
		# defensively so a refactor of cdp_use's internals doesn't break
		# this method.
		state = getattr(ws, 'state', None)
		return bool(state) and str(state).upper().endswith('OPEN')


# ── BiDi (Playwright) backend ────────────────────────────────────────────────


class BidiBrowserConnection(BrowserConnection):
	"""Wraps a Playwright :class:`Browser` reached over BiDi WebSocket.

	Construction takes EITHER an already-running ``Browser`` (when the
	engine launched it standalone via :meth:`FirefoxPlaywrightEngine.launch_with_adapter`)
	OR a ``ws_endpoint`` that this connection will :meth:`connect` to on
	:meth:`start`. The class hides which path was used downstream.

	Two-step lifecycle:

	1. ``__init__`` records inputs but does NOT touch the network.
	2. :meth:`start` enters async-land: starts the Playwright runtime
	   if not already running, connects to ``ws_endpoint`` if given,
	   stores the resulting ``Browser`` handle.
	"""

	backend = 'bidi'

	def __init__(
		self,
		*,
		browser: PlaywrightBrowser | None = None,
		playwright: Playwright | None = None,
		ws_endpoint: str | None = None,
	) -> None:
		if browser is None and ws_endpoint is None:
			raise ValueError(
				'BidiBrowserConnection requires either `browser` (already-'
				'connected Playwright Browser) or `ws_endpoint` to connect to'
			)
		self._browser: PlaywrightBrowser | None = browser
		self._playwright: Playwright | None = playwright
		self._ws_endpoint = ws_endpoint
		self._owns_playwright = playwright is None and browser is None
		self._started = browser is not None

	@property
	def browser(self) -> PlaywrightBrowser:
		"""Access the underlying Playwright Browser. Raises if not started."""
		if self._browser is None:
			raise RuntimeError('BidiBrowserConnection: start() not called yet')
		return self._browser

	@property
	def playwright(self) -> Playwright | None:
		"""The Playwright runtime — ``None`` when the caller injected an
		already-connected Browser and kept the runtime under its own
		control."""
		return self._playwright

	async def start(self) -> None:
		if self._started and self._browser is not None:
			return
		from playwright.async_api import async_playwright

		if self._playwright is None:
			self._playwright = await async_playwright().start()
		assert self._ws_endpoint is not None, 'ws_endpoint required when no Browser injected'
		self._browser = await self._playwright.firefox.connect(self._ws_endpoint)
		self._started = True

	async def stop(self) -> None:
		# Order: browser → playwright runtime. Skip what we don't own.
		if self._browser is not None:
			try:
				await self._browser.close()
			except Exception:
				pass
			self._browser = None
		if self._owns_playwright and self._playwright is not None:
			try:
				await self._playwright.stop()
			except Exception:
				pass
			self._playwright = None
		self._started = False

	@property
	def is_open(self) -> bool:
		if not self._started or self._browser is None:
			return False
		# Playwright Browser exposes `is_connected()` for liveness.
		return bool(self._browser.is_connected())


# ── Factory helpers ──────────────────────────────────────────────────────────


def connection_from_browser_type(browser_type: str) -> type[BrowserConnection]:
	"""Resolve a :class:`~browser_use.browser.profile.BrowserType`-shaped
	string (``'chromium'`` / ``'firefox'``) to the connection class that
	wraps that backend's control protocol."""
	if browser_type == 'chromium':
		return CdpBrowserConnection
	if browser_type == 'firefox':
		return BidiBrowserConnection
	raise ValueError(f'no BrowserConnection mapped for browser_type={browser_type!r}')


__all__ = [
	'BrowserConnection',
	'CdpBrowserConnection',
	'BidiBrowserConnection',
	'connection_from_browser_type',
]
