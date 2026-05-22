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

	def __init__(self, session: BrowserSession) -> None:
		self._session = session

	# Helpers ----------------------------------------------------------------

	async def _eval(self, expression: str, *, return_by_value: bool = True) -> Any:
		"""``Runtime.evaluate`` round-trip. Returns the unwrapped value."""
		cdp = self._session.cdp_client
		# Pick the current target session — BrowserSession owns the routing
		# rules (active tab, frame). ``Runtime.evaluate`` runs in the
		# session's main world.
		session_id = self._session.current_session_id
		result = await cdp.send.Runtime.evaluate(
			params={'expression': expression, 'returnByValue': return_by_value,
			        'awaitPromise': True},
			session_id=session_id,
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
		session_id = self._session.current_session_id
		await cdp.send.Page.navigate(
			params={'url': url, 'transitionType': 'typed'},
			session_id=session_id,
		)
		await self.wait_for_load_state(wait_until, timeout_ms=timeout_ms)
		return {'url': await self.url(), 'status': None}  # CDP doesn't surface main-frame status here

	async def reload(self, *, wait_until: str = 'load',
	                 timeout_ms: int = 30_000) -> None:
		cdp = self._session.cdp_client
		await cdp.send.Page.reload(params={}, session_id=self._session.current_session_id)
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
			session_id=self._session.current_session_id,
		)

	# Media ------------------------------------------------------------------

	async def screenshot(self, *, full_page: bool = False,
	                     fmt: Literal['png', 'jpeg'] = 'png') -> bytes:
		cdp = self._session.cdp_client
		result = await cdp.send.Page.captureScreenshot(
			params={'format': fmt, 'captureBeyondViewport': full_page},
			session_id=self._session.current_session_id,
		)
		return base64.b64decode(result['data'])

	# Accessibility ----------------------------------------------------------

	async def accessibility_snapshot(self, *,
	                                 interesting_only: bool = True) -> dict | None:
		cdp = self._session.cdp_client
		try:
			result = await cdp.send.Accessibility.getFullAXTree(
				params={}, session_id=self._session.current_session_id,
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
			params=params, session_id=self._session.current_session_id,
		)
		return result.get('cookies', [])

	async def set_cookies(self, cookies: list[dict[str, Any]]) -> None:
		cdp = self._session.cdp_client
		await cdp.send.Network.setCookies(
			params={'cookies': cookies}, session_id=self._session.current_session_id,
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
