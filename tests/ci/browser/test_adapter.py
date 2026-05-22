"""Adapter contract tests.

Two halves:

  - ``test_abc_*`` — verify that the abstract contract refuses partial
    implementations and that every concrete adapter in this module
    exposes every method on the ABC. These tests don't launch a
    browser; they're pure interface checks.

  - ``test_playwright_*`` — drive the Playwright adapter against a
    pytest-httpserver fixture (no real URLs, no mocks) to verify the
    page-level operations actually do what the ABC promises.

The CDP adapter needs a running ``BrowserSession`` against a real
Chromium subprocess to exercise. Those tests live alongside the rest
of the CDP-flavoured suite and aren't duplicated here — the ABC-level
checks below already lock the surface.
"""

from __future__ import annotations

import inspect
import pytest

from browser_use.browser.adapter import (
	BrowserAdapter,
	CdpBrowserAdapter,
	PlaywrightBrowserAdapter,
)


# ── ABC contract ────────────────────────────────────────────────────────────


def test_browser_adapter_is_abstract() -> None:
	"""You can't instantiate the bare ABC — guards against accidental
	use of the placeholder contract."""
	with pytest.raises(TypeError):
		BrowserAdapter()  # type: ignore[abstract]


def test_concrete_adapters_implement_every_abstract_method() -> None:
	"""Every method declared abstract on BrowserAdapter must be
	concretely implemented on every shipped adapter. This is what
	makes the abstraction load-bearing — consumers should never need
	to check the adapter's type before calling a method."""
	abstract_methods = {
		name for name, method in BrowserAdapter.__dict__.items()
		if getattr(method, '__isabstractmethod__', False)
	}
	assert abstract_methods, 'BrowserAdapter should have abstract methods'

	for adapter_cls in (CdpBrowserAdapter, PlaywrightBrowserAdapter):
		assert not adapter_cls.__abstractmethods__, (
			f'{adapter_cls.__name__} is missing concrete impls for '
			f'{adapter_cls.__abstractmethods__}'
		)
		# Belt-and-braces: every abstract name must resolve to a real
		# coroutine function on the subclass.
		for name in abstract_methods:
			fn = getattr(adapter_cls, name, None)
			assert fn is not None, f'{adapter_cls.__name__} is missing {name}'
			assert inspect.iscoroutinefunction(fn), (
				f'{adapter_cls.__name__}.{name} should be `async def`'
			)


def test_concrete_adapter_signatures_match_abc() -> None:
	"""Parameter names and types stay aligned across implementations.
	Consumers pass kwargs (timeout_ms=…) — if an adapter renames a
	parameter, those calls silently break. Lock the signatures here."""
	for name, abc_method in BrowserAdapter.__dict__.items():
		if not getattr(abc_method, '__isabstractmethod__', False):
			continue
		abc_sig = inspect.signature(abc_method)
		for adapter_cls in (CdpBrowserAdapter, PlaywrightBrowserAdapter):
			impl_sig = inspect.signature(getattr(adapter_cls, name))
			assert list(impl_sig.parameters) == list(abc_sig.parameters), (
				f'{adapter_cls.__name__}.{name} param order diverges from ABC: '
				f'{list(impl_sig.parameters)} vs {list(abc_sig.parameters)}'
			)


# ── Playwright adapter end-to-end ───────────────────────────────────────────
#
# These run only when Playwright is installed AND a browser binary is
# available (the CI image installs Chromium by default; Firefox is
# opt-in). Skip cleanly otherwise so the suite stays green on minimal
# environments.


@pytest.fixture
def playwright_page():
	"""A real Playwright Chromium page. Skip if Playwright isn't
	available — we don't want the CI suite to require all backends."""
	try:
		from playwright.async_api import async_playwright
	except ImportError:
		pytest.skip('playwright not installed')

	import asyncio
	loop = asyncio.get_event_loop()

	pw = loop.run_until_complete(async_playwright().start())
	try:
		browser = loop.run_until_complete(pw.chromium.launch(headless=True))
	except Exception as e:
		loop.run_until_complete(pw.stop())
		pytest.skip(f'no Chromium browser available: {e}')
	context = loop.run_until_complete(browser.new_context())
	page = loop.run_until_complete(context.new_page())
	try:
		yield page
	finally:
		loop.run_until_complete(page.close())
		loop.run_until_complete(context.close())
		loop.run_until_complete(browser.close())
		loop.run_until_complete(pw.stop())


async def test_playwright_adapter_goto_and_content(playwright_page, httpserver) -> None:
	httpserver.expect_request('/index').respond_with_data(
		'<html><head><title>OK</title></head><body><h1 class="g">Hi</h1></body></html>',
		content_type='text/html',
	)
	adapter = PlaywrightBrowserAdapter(playwright_page)
	result = await adapter.goto(httpserver.url_for('/index'))
	assert result['status'] == 200
	html = await adapter.content()
	assert 'Hi' in html
	assert await adapter.title() == 'OK'


async def test_playwright_adapter_locator(playwright_page, httpserver) -> None:
	httpserver.expect_request('/list').respond_with_data(
		'<ul><li class="x">a</li><li class="x">b</li><li class="x">c</li></ul>',
		content_type='text/html',
	)
	adapter = PlaywrightBrowserAdapter(playwright_page)
	await adapter.goto(httpserver.url_for('/list'))
	assert await adapter.locator_count('li.x') == 3
	assert await adapter.locator_inner_text('li.x') == 'a'
	assert await adapter.locator_get_attribute('li.x', 'class') == 'x'
	assert await adapter.locator_is_visible('li.x') is True
	assert await adapter.locator_count('li.missing') == 0
	assert await adapter.locator_is_visible('li.missing') is False


async def test_playwright_adapter_evaluate(playwright_page, httpserver) -> None:
	httpserver.expect_request('/blank').respond_with_data('<html><body></body></html>',
	                                                       content_type='text/html')
	adapter = PlaywrightBrowserAdapter(playwright_page)
	await adapter.goto(httpserver.url_for('/blank'))
	assert await adapter.evaluate('1 + 1') == 2
	# Function-with-arg form (Playwright convention).
	assert await adapter.evaluate('(n) => n * 7', 6) == 42
