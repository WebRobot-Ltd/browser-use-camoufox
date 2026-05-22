"""Screenshot watchdog — dual-mode CDP / Playwright via BrowserAdapter."""

import base64
from typing import TYPE_CHECKING, Any, ClassVar

from bubus import BaseEvent
from cdp_use.cdp.page import CaptureScreenshotParameters

from browser_use.browser.events import ScreenshotEvent
from browser_use.browser.views import BrowserError
from browser_use.browser.watchdog_base import BaseWatchdog
from browser_use.observability import observe_debug

if TYPE_CHECKING:
	pass


class ScreenshotWatchdog(BaseWatchdog):
	"""Handles screenshot requests on both CDP and BiDi backends.

	Phase-5b port. The Chromium path (CDP) is bit-identical to the
	previous implementation. On BiDi (Firefox/Camoufox) we go through
	the :class:`PlaywrightBrowserAdapter` which has the
	:meth:`screenshot` method on the page-level contract.

	Caveat for the BiDi path: ``event.clip`` is honoured only on the
	CDP backend today — Playwright's ``page.screenshot(clip=…)`` exists
	but uses a slightly different coordinate origin and we haven't
	verified parity yet. When ``event.clip`` is set on BiDi we still
	take a full-page / full-viewport screenshot and log a warning.
	"""

	# Events this watchdog listens to
	LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [ScreenshotEvent]

	# Events this watchdog emits
	EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

	@observe_debug(ignore_input=True, ignore_output=True, name='screenshot_event_handler')
	async def on_ScreenshotEvent(self, event: ScreenshotEvent) -> str:
		"""Handle screenshot request. Dispatches by backend.

		Returns:
			Base64-encoded PNG screenshot data (no `data:` URL prefix).
		"""
		# Phase-5b dispatch: route to the BiDi handler when the session
		# is on a Playwright/Firefox backend. Falls through to the
		# legacy CDP path otherwise.
		conn = getattr(self.browser_session, '_connection', None)
		if conn is not None and conn.backend == 'bidi':
			return await self._on_screenshot_bidi(event)
		return await self._on_screenshot_cdp(event)

	async def _on_screenshot_cdp(self, event: ScreenshotEvent) -> str:
		"""Legacy CDP path — bit-identical to the pre-Phase-5b implementation."""
		self.logger.debug('[ScreenshotWatchdog] (CDP) Handler START - on_ScreenshotEvent called')
		try:
			# Validate focused target is a top-level page (not iframe/worker)
			# CDP Page.captureScreenshot only works on page/tab targets
			focused_target = self.browser_session.get_focused_target()

			if focused_target and focused_target.target_type in ('page', 'tab'):
				target_id = focused_target.target_id
			else:
				# Focused target is iframe/worker/missing - fall back to any page target
				target_type_str = focused_target.target_type if focused_target else 'None'
				self.logger.warning(f'[ScreenshotWatchdog] Focused target is {target_type_str}, falling back to page target')
				page_targets = self.browser_session.get_page_targets()
				if not page_targets:
					raise BrowserError('[ScreenshotWatchdog] No page targets available for screenshot')
				target_id = page_targets[-1].target_id

			cdp_session = await self.browser_session.get_or_create_cdp_session(target_id, focus=True)

			# Remove highlights BEFORE taking the screenshot so they don't appear in the image.
			# Done here (not in finally) so CancelledError is never swallowed — any await in a
			# finally block can suppress external task cancellation.
			# remove_highlights() has its own asyncio.timeout(3.0) internally so it won't block.
			try:
				await self.browser_session.remove_highlights()
			except Exception:
				pass

			# Prepare screenshot parameters
			params_dict: dict[str, Any] = {'format': 'png', 'captureBeyondViewport': event.full_page}
			if event.clip:
				params_dict['clip'] = {
					'x': event.clip['x'],
					'y': event.clip['y'],
					'width': event.clip['width'],
					'height': event.clip['height'],
					'scale': 1,
				}
			params = CaptureScreenshotParameters(**params_dict)

			# Take screenshot using CDP
			self.logger.debug(f'[ScreenshotWatchdog] (CDP) Taking screenshot with params: {params}')
			result = await cdp_session.cdp_client.send.Page.captureScreenshot(params=params, session_id=cdp_session.session_id)

			# Return base64-encoded screenshot data
			if result and 'data' in result:
				self.logger.debug('[ScreenshotWatchdog] (CDP) Screenshot captured successfully')
				return result['data']

			raise BrowserError('[ScreenshotWatchdog] (CDP) Screenshot result missing data')
		except Exception as e:
			self.logger.error(f'[ScreenshotWatchdog] (CDP) Screenshot failed: {e}')
			raise

	async def _on_screenshot_bidi(self, event: ScreenshotEvent) -> str:
		"""BiDi (Playwright Firefox/Camoufox) path via BrowserAdapter."""
		self.logger.debug('[ScreenshotWatchdog] (BiDi) Handler START - on_ScreenshotEvent called')
		try:
			# Try removing highlights first — same posture as the CDP path.
			# The method is CDP-bound on the current codebase and no-ops
			# (or raises silently) on BiDi; suppress and continue.
			try:
				await self.browser_session.remove_highlights()
			except Exception:
				pass

			if event.clip:
				self.logger.warning(
					'[ScreenshotWatchdog] (BiDi) `event.clip` not yet honoured on the '
					'Playwright path — taking a full-viewport screenshot instead'
				)

			adapter = await self.browser_session.get_adapter()
			png_bytes = await adapter.screenshot(full_page=bool(event.full_page), fmt='png')
			# Return base64-encoded PNG (no data: prefix) to match the CDP path's contract.
			b64 = base64.b64encode(png_bytes).decode('ascii')
			self.logger.debug('[ScreenshotWatchdog] (BiDi) Screenshot captured successfully')
			return b64
		except Exception as e:
			self.logger.error(f'[ScreenshotWatchdog] (BiDi) Screenshot failed: {e}')
			raise
