"""Browser engine abstraction — the seam between launch and protocol layers.

browser-use today drives Chromium-family browsers via raw CDP (the cdp_use
library). To support Firefox / Camoufox we need a second engine that
speaks WebDriver BiDi + Marionette via Playwright Firefox. Those two
protocol stacks share almost nothing, so they have to live behind a
common interface.

This module defines:

  - :class:`BrowserEngine` — the abstract contract every engine implements.
  - :class:`ChromiumCdpEngine` — wraps the existing CDP-based launch path
    by delegating back to :class:`~browser_use.browser.watchdogs.local_browser_watchdog.LocalBrowserWatchdog`'s
    private helper. No behaviour change vs. previous releases.
  - :class:`FirefoxPlaywrightEngine` — real Playwright Firefox launcher.
    Spawns a Firefox process (stock Firefox or a Camoufox binary via
    ``executable_path``) and surfaces the BiDi WebSocket URL via
    :meth:`launch`. For local tests and adapter-driven scripts use
    :meth:`launch_with_adapter` — it returns a Playwright page already
    wrapped in :class:`~browser_use.browser.adapter.PlaywrightBrowserAdapter`.
  - :func:`get_engine` — factory that resolves :class:`~browser_use.browser.profile.BrowserType`
    into a concrete engine.

Status note. The Firefox **launch** is wired and works; the Firefox
**session** path is NOT wired into :class:`LocalBrowserWatchdog` yet,
because :mod:`browser_use.browser.session` only speaks CDP today —
the BiDi WS URL the engine returns is non-consumable by the CDP
session bring-up. That's the Phase-5 split. Until then, Firefox is
usable via :meth:`FirefoxPlaywrightEngine.launch_with_adapter` (and
through the BrowserToolActor in the agentic-runtime, which never
went through the watchdog in the first place).

Quick local test:

    from browser_use.browser.engine import FirefoxPlaywrightEngine

    handle = await FirefoxPlaywrightEngine.launch_with_adapter(headless=True)
    try:
        await handle['adapter'].goto('https://books.toscrape.com')
        html = await handle['adapter'].content()
        print(html[:200])
    finally:
        await handle['teardown']()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import psutil

from browser_use.browser.profile import BrowserType


if TYPE_CHECKING:
	from browser_use.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog


class BrowserEngine(ABC):
	"""The contract every browser engine implements.

	An *engine* owns the rules for bringing a browser subprocess up and
	exposing whatever the protocol layer needs to connect. For
	:class:`ChromiumCdpEngine` that's a process + a CDP-over-HTTP URL; for
	:class:`FirefoxPlaywrightEngine` it will be a process + a BiDi
	WebSocket URL. We keep the return shape compatible across engines for
	now (``(psutil.Process, ws_url)``) because the session bring-up code
	only needs the URL — what protocol speaks on the other side is the
	engine's problem.
	"""

	#: Stable short identifier — handy for log lines.
	name: str = 'base'

	@abstractmethod
	async def launch(self, watchdog: LocalBrowserWatchdog) -> tuple[psutil.Process, str]:
		"""Launch a browser subprocess and return ``(process, ws_url)``.

		``ws_url`` is whatever URL the session layer must connect to. For
		Chromium engines this is the CDP HTTP+WS endpoint discovered via
		``http://localhost:<port>/json/version``. For Firefox engines this
		will be the BiDi WebSocket URL printed by ``geckodriver``.
		"""
		raise NotImplementedError


class ChromiumCdpEngine(BrowserEngine):
	"""Chromium-family launch path — Chrome, Chromium, Edge, channels thereof.

	Today this is a thin pass-through to
	:meth:`LocalBrowserWatchdog._launch_browser_chromium_impl`, which still
	owns the real subprocess + temp-dir + retry logic. Once that logic is
	extracted into this class (Phase 3), the delegation goes away.
	"""

	name = 'chromium'

	async def launch(self, watchdog: LocalBrowserWatchdog) -> tuple[psutil.Process, str]:
		return await watchdog._launch_browser_chromium_impl()


class FirefoxPlaywrightEngine(BrowserEngine):
	"""Firefox-family launch path — stock Firefox + Camoufox.

	Spawns a Firefox subprocess via Playwright Python's
	``firefox.launch_server`` (for the ABC :meth:`launch` contract that
	returns a connectable URL) or ``firefox.launch`` (for the
	:meth:`launch_with_adapter` convenience that returns a wrapped Page).

	The Camoufox binary is a Firefox build — drop its path into
	``profile.executable_path`` (or the ``executable_path=`` keyword)
	and the engine launches Camoufox transparently.

	**Session integration is not wired yet.** The BiDi WebSocket URL
	returned by :meth:`launch` is the right protocol for any client
	speaking Playwright's wire format, but :mod:`browser_use.browser.session`
	is CDP-only and can't consume it. Going through
	:class:`LocalBrowserWatchdog` on a FIREFOX profile would launch the
	browser successfully then fail to connect downstream. That's
	Phase-5. Until then, use :meth:`launch_with_adapter` for any code
	that needs to drive a Firefox/Camoufox page through
	:class:`~browser_use.browser.adapter.PlaywrightBrowserAdapter`
	without the legacy session stack.
	"""

	name = 'firefox'

	async def launch(self, watchdog: LocalBrowserWatchdog) -> tuple[psutil.Process, str]:
		"""ABC-conformant launch. Reads launch flags from the
		watchdog's :class:`~browser_use.browser.profile.BrowserProfile`
		and returns ``(process, ws_endpoint)`` — a BiDi WebSocket URL a
		Playwright client can ``connect()`` to.
		"""
		from playwright.async_api import async_playwright

		profile = watchdog.browser_session.browser_profile
		pw = await async_playwright().start()
		server = await pw.firefox.launch_server(
			headless=bool(profile.headless),
			executable_path=str(profile.executable_path) if profile.executable_path else None,
			args=list(profile.args or []),
		)
		pid = getattr(server, 'process', None)
		pid = pid.pid if pid is not None else None
		if pid is None:
			# Playwright should always surface the subprocess pid, but
			# defend against the corner case so the upper layer gets a
			# clean error instead of an AttributeError surfaced from
			# `psutil.Process(None)`.
			raise RuntimeError('playwright.firefox.launch_server did not expose a process pid')
		return psutil.Process(pid), server.ws_endpoint

	@staticmethod
	async def launch_with_adapter(
		*,
		headless: bool = True,
		executable_path: str | None = None,
		args: list[str] | None = None,
		proxy: dict | None = None,
	) -> dict:
		"""End-to-end local launch: spawn Firefox, open a context + page,
		wrap the page in :class:`PlaywrightBrowserAdapter`, return a
		dict with everything callers need plus a teardown coroutine.

		Returned shape::

		    {
		      'process':  psutil.Process | None,
		      'browser':  playwright.async_api.Browser,
		      'page':     playwright.async_api.Page,
		      'adapter':  PlaywrightBrowserAdapter,
		      'teardown': Callable[[], Awaitable[None]],
		    }

		The caller must ``await handle['teardown']()`` to release the
		browser + playwright runtime. Designed for scripts, notebooks,
		and the kind of local-test we use to validate the fork against
		Camoufox before plumbing into the full Agent loop.
		"""
		from playwright.async_api import async_playwright

		from browser_use.browser.adapter import PlaywrightBrowserAdapter

		pw = await async_playwright().start()
		browser = await pw.firefox.launch(
			headless=headless,
			executable_path=executable_path,
			args=args or [],
			proxy=proxy,
		)
		context = await browser.new_context()
		page = await context.new_page()

		# Playwright's Browser exposes `.process` as the asyncio subprocess
		# handle on local launches (None for `connect()`-ed remotes).
		raw_proc = getattr(browser, 'process', None)
		process = psutil.Process(raw_proc.pid) if raw_proc is not None else None
		adapter = PlaywrightBrowserAdapter(page)

		async def _teardown() -> None:
			# Order matters: close page → context → browser → pw runtime.
			# Each step is best-effort; we never raise from teardown.
			for step in (page.close, context.close, browser.close, pw.stop):
				try:
					await step()
				except Exception:
					pass

		return {
			'process':  process,
			'browser':  browser,
			'page':     page,
			'adapter':  adapter,
			'teardown': _teardown,
		}


# ── Factory ──────────────────────────────────────────────────────────────────


_ENGINES: dict[BrowserType, BrowserEngine] = {
	BrowserType.CHROMIUM: ChromiumCdpEngine(),
	BrowserType.FIREFOX: FirefoxPlaywrightEngine(),
}


def get_engine(browser_type: BrowserType) -> BrowserEngine:
	"""Return the engine matching ``browser_type``.

	Engines are stateless and shared across sessions — they hold no
	per-launch state, only the dispatch rules for that engine family.
	"""
	try:
		return _ENGINES[browser_type]
	except KeyError as e:
		raise ValueError(f'unknown browser_type: {browser_type!r}') from e
