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
  - :class:`FirefoxPlaywrightEngine` — STUB. Raises :class:`NotImplementedError`
    with a useful message until the Firefox launch + session paths land.
  - :func:`get_engine` — factory that resolves :class:`~browser_use.browser.profile.BrowserType`
    into a concrete engine.

The session and watchdog code stays CDP-end-to-end for the CHROMIUM path
in this phase; Phase-3 will start extracting the actual Chromium launch
logic out of :class:`LocalBrowserWatchdog` into :class:`ChromiumCdpEngine`,
and Phase-4 will plug Playwright Firefox into :class:`FirefoxPlaywrightEngine`.
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

	**Status: not implemented.** Phase-4 will land Playwright Firefox
	launching here (subprocess via ``playwright.firefox.launch_server()``,
	BiDi WebSocket URL surfaced to the session layer). For now this engine
	exists so that the factory has something to return and config
	validation can reject impossible combinations early instead of failing
	deep inside a CDP call.
	"""

	name = 'firefox'

	async def launch(self, watchdog: LocalBrowserWatchdog) -> tuple[psutil.Process, str]:
		raise NotImplementedError(
			'BrowserType.FIREFOX is not yet wired into the launch path. '
			'The data model accepts it (see BrowserProfile.browser_type), '
			'but the session/watchdog stack still assumes CDP end-to-end. '
			"Track progress on the `firefox-compat` branch. For an immediate "
			'Camoufox-driven selector-grounding workflow, use the '
			'BrowserToolActor in the webrobot-etl-chatbot-server agentic-runtime '
			'(thin Playwright-Firefox loop, not a full browser-use port).'
		)


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
