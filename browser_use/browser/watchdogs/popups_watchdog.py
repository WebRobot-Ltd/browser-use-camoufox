"""Watchdog for handling JavaScript dialogs (alert, confirm, prompt) automatically."""

import asyncio
from typing import ClassVar

from bubus import BaseEvent
from pydantic import PrivateAttr

from browser_use.browser.events import TabCreatedEvent
from browser_use.browser.watchdog_base import BaseWatchdog


class PopupsWatchdog(BaseWatchdog):
	"""Handles JavaScript dialogs (alert, confirm, prompt) by automatically accepting them immediately."""

	# Events this watchdog listens to and emits
	LISTENS_TO: ClassVar[list[type[BaseEvent]]] = [TabCreatedEvent]
	EMITS: ClassVar[list[type[BaseEvent]]] = []

	# Track which targets have dialog handlers registered
	_dialog_listeners_registered: set[str] = PrivateAttr(default_factory=set)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.logger.debug(f'🚀 PopupsWatchdog initialized with browser_session={self.browser_session}, ID={id(self)}')

	async def on_TabCreatedEvent(self, event: TabCreatedEvent) -> None:
		"""Set up JavaScript dialog handling when a new tab is created.

		Dual-mode dispatch (Phase-5b): on the BiDi backend we attach a
		Playwright ``page.on('dialog', …)`` handler to the connection's
		current page instead of CDP's ``Page.javascriptDialogOpening``
		registration. Same operational contract — JS alerts/confirms get
		auto-accepted, prompts dismissed, beforeunload accepted.
		"""
		conn = getattr(self.browser_session, '_connection', None)
		if conn is not None and conn.backend == 'bidi':
			await self._on_tab_created_bidi()
			return

		target_id = event.target_id
		self.logger.debug(f'🎯 PopupsWatchdog received TabCreatedEvent for target {target_id}')

		# Skip if we've already registered for this target
		if target_id in self._dialog_listeners_registered:
			self.logger.debug(f'Already registered dialog handlers for target {target_id}')
			return

		self.logger.debug(f'📌 Starting dialog handler setup for target {target_id}')
		try:
			# Get all CDP sessions for this target and any child frames
			cdp_session = await self.browser_session.get_or_create_cdp_session(
				target_id, focus=False
			)  # don't auto-focus new tabs! sometimes we need to open tabs in background

			# CRITICAL: Enable Page domain to receive dialog events
			try:
				await cdp_session.cdp_client.send.Page.enable(session_id=cdp_session.session_id)
				self.logger.debug(f'✅ Enabled Page domain for session {cdp_session.session_id[-8:]}')
			except Exception as e:
				self.logger.debug(f'Failed to enable Page domain: {e}')

			# Also register for the root CDP client to catch dialogs from any frame
			if self.browser_session._cdp_client_root:
				self.logger.debug('📌 Also registering handler on root CDP client')
				try:
					# Enable Page domain on root client too
					await self.browser_session._cdp_client_root.send.Page.enable()
					self.logger.debug('✅ Enabled Page domain on root CDP client')
				except Exception as e:
					self.logger.debug(f'Failed to enable Page domain on root: {e}')

			# Set up async handler for JavaScript dialogs - accept immediately without event dispatch
			async def handle_dialog(event_data, session_id: str | None = None):
				"""Handle JavaScript dialog events - accept immediately."""
				try:
					dialog_type = event_data.get('type', 'alert')
					message = event_data.get('message', '')

					# Store the popup message in browser session for inclusion in browser state
					if message:
						formatted_message = f'[{dialog_type}] {message}'
						self.browser_session._closed_popup_messages.append(formatted_message)
						self.logger.debug(f'📝 Stored popup message: {formatted_message[:100]}')

					# Choose action based on dialog type:
					# - alert: accept=true (click OK to dismiss)
					# - confirm: accept=true (click OK to proceed - safer for automation)
					# - prompt: accept=false (click Cancel since we can't provide input)
					# - beforeunload: accept=true (allow navigation)
					should_accept = dialog_type in ('alert', 'confirm', 'beforeunload')

					action_str = 'accepting (OK)' if should_accept else 'dismissing (Cancel)'
					self.logger.info(f"🔔 JavaScript {dialog_type} dialog: '{message[:100]}' - {action_str}...")

					dismissed = False

					# Approach 1: Use the session that detected the dialog (most reliable)
					if self.browser_session._cdp_client_root and session_id:
						try:
							self.logger.debug(f'🔄 Approach 1: Using detecting session {session_id[-8:]}')
							await asyncio.wait_for(
								self.browser_session._cdp_client_root.send.Page.handleJavaScriptDialog(
									params={'accept': should_accept},
									session_id=session_id,
								),
								timeout=0.5,
							)
							dismissed = True
							self.logger.info('✅ Dialog handled successfully via detecting session')
						except (TimeoutError, Exception) as e:
							self.logger.debug(f'Approach 1 failed: {type(e).__name__}')

					# Approach 2: Try with current agent focus session
					if not dismissed and self.browser_session._cdp_client_root and self.browser_session.agent_focus_target_id:
						try:
							# Use public API with focus=False to avoid changing focus during popup dismissal
							cdp_session = await self.browser_session.get_or_create_cdp_session(
								self.browser_session.agent_focus_target_id, focus=False
							)
							self.logger.debug(f'🔄 Approach 2: Using agent focus session {cdp_session.session_id[-8:]}')
							await asyncio.wait_for(
								self.browser_session._cdp_client_root.send.Page.handleJavaScriptDialog(
									params={'accept': should_accept},
									session_id=cdp_session.session_id,
								),
								timeout=0.5,
							)
							dismissed = True
							self.logger.info('✅ Dialog handled successfully via agent focus session')
						except (TimeoutError, Exception) as e:
							self.logger.debug(f'Approach 2 failed: {type(e).__name__}')

				except Exception as e:
					self.logger.error(f'❌ Critical error in dialog handler: {type(e).__name__}: {e}')

			# Register handler on the specific session
			cdp_session.cdp_client.register.Page.javascriptDialogOpening(handle_dialog)  # type: ignore[arg-type]
			self.logger.debug(
				f'Successfully registered Page.javascriptDialogOpening handler for session {cdp_session.session_id}'
			)

			# Also register on root CDP client to catch dialogs from any frame
			if hasattr(self.browser_session._cdp_client_root, 'register'):
				try:
					self.browser_session._cdp_client_root.register.Page.javascriptDialogOpening(handle_dialog)  # type: ignore[arg-type]
					self.logger.debug('Successfully registered dialog handler on root CDP client for all frames')
				except Exception as root_error:
					self.logger.warning(f'Failed to register on root CDP client: {root_error}')

			# Mark this target as having dialog handling set up
			self._dialog_listeners_registered.add(target_id)

			self.logger.debug(f'Set up JavaScript dialog handling for tab {target_id}')

		except Exception as e:
			self.logger.warning(f'Failed to set up popup handling for tab {target_id}: {e}')

	# ── BiDi (Playwright) helpers ───────────────────────────────────────────

	async def _on_tab_created_bidi(self) -> None:
		"""Register a Playwright ``page.on('dialog', …)`` handler.

		Playwright auto-binds the handler to the page; once attached it
		fires for every alert/confirm/prompt/beforeunload until the
		page is closed. Idempotent: re-attaching to a page that already
		has our handler is a no-op (we tag the page object).
		"""
		conn = self.browser_session._connection
		try:
			page = conn.current_page
		except Exception as e:
			self.logger.debug(f'[PopupsWatchdog] (BiDi) no current page yet: {e}')
			return

		# Tag the page so we don't double-register if TabCreatedEvent
		# fires more than once for the same page.
		marker = '__popups_watchdog_attached__'
		if getattr(page, marker, False):
			return

		async def _handle(dialog) -> None:
			try:
				dialog_type = dialog.type
				message = dialog.message or ''
				if message:
					formatted = f'[{dialog_type}] {message}'
					self.browser_session._closed_popup_messages.append(formatted)
				# Same semantics as the CDP path: accept alert / confirm /
				# beforeunload, dismiss prompt.
				if dialog_type in ('alert', 'confirm', 'beforeunload'):
					await dialog.accept()
					self.logger.info(f"🔔 (BiDi) {dialog_type} dialog: '{message[:100]}' - accepted")
				else:
					await dialog.dismiss()
					self.logger.info(f"🔔 (BiDi) {dialog_type} dialog: '{message[:100]}' - dismissed")
			except Exception as e:
				self.logger.error(f'[PopupsWatchdog] (BiDi) handler error: {type(e).__name__}: {e}')

		page.on('dialog', _handle)
		try:
			setattr(page, marker, True)
		except Exception:
			pass  # some Playwright page implementations are slot-bound
		self.logger.debug('[PopupsWatchdog] (BiDi) page.on("dialog") registered')
