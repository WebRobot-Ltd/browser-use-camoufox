# Writing a new BrowserAdapter

The `BrowserAdapter` abstraction in [`browser_use/browser/adapter.py`](../browser_use/browser/adapter.py) is the convergence point between the original CDP-based codebase and any new browser engine we want to support — today Playwright Firefox / Camoufox, tomorrow maybe WebKit, BiDi-native drivers, remote browser-as-a-service products, headless replay rigs, etc.

This document is the recipe for adding a third (or fourth) backend so the agentic loop keeps working unchanged.

## The contract in one paragraph

A `BrowserAdapter` is a **page-level** controller. It exposes the operations a consumer needs to drive *a single page* — navigate, evaluate JS, query elements by CSS, click/fill/press, snapshot accessibility, screenshot. The contract is intentionally a strict subset of Playwright's async `Page` API (timeouts in milliseconds, CSS selectors, async everywhere). Anything that's *not* in that contract is either a launcher concern (see [`browser_use/browser/engine.py`](../browser_use/browser/engine.py)) or a backend-specific feature you should think twice about leaking into the abstraction.

## When to add a new adapter

Add a new adapter when you need to drive a browser via a control protocol the existing adapters can't speak. Concretely:

| Backend | Existing adapter? | Notes |
| --- | --- | --- |
| Chrome / Chromium / Edge via CDP | `CdpBrowserAdapter` | The original codebase |
| Stock Firefox via Playwright | `PlaywrightBrowserAdapter` | Just pass a `playwright.firefox.launch().new_page()` |
| Camoufox (anti-detect Firefox fork) | `PlaywrightBrowserAdapter` | Same — Camoufox launches a Playwright-driven Firefox |
| Stock Firefox via Marionette/BiDi (no Playwright) | none yet → **new adapter** | Direct WebDriver BiDi, smaller dep footprint |
| Remote BaaS (Browserless, Steel, Hyperbrowser, Brightdata Browser API) | none yet → **new adapter** | Each speaks its own protocol on top of CDP/Playwright |
| WebKit (Playwright) | `PlaywrightBrowserAdapter` works in theory | The page contract is shared; no separate adapter needed |
| Headless replay (record/replay over saved HAR) | none yet → **new adapter** | All ops resolved against the static snapshot |

**Don't add an adapter when the difference is purely launch-time** (e.g. "Chrome with proxy X" vs "Chrome with proxy Y"). That's `BrowserEngine` / `BrowserProfile` territory.

## Steps

### 1. Inherit `BrowserAdapter`

```python
from browser_use.browser.adapter import BrowserAdapter


class MyXBrowserAdapter(BrowserAdapter):
    """One-line summary of what backend this adapter wraps."""

    def __init__(self, handle):
        # `handle` is whatever your backend gave you to talk to the page.
        # Don't take a profile or session — the adapter is operations-only.
        # The caller built the connection; you wrap it.
        self._handle = handle
```

The class-level docstring should answer: *what does this adapter wrap* (a session object, a Playwright Page, an HTTP client), *what does it own* (typically nothing — caller owns the lifecycle), and *what's the operational mode* (real browser, replay, BaaS).

### 2. Implement every method

There are no optional methods. Every method on `BrowserAdapter` is abstract — Python will refuse to instantiate your class if you skip one. This is intentional: consumers should be able to use *any* method on *any* adapter without surprise.

If your backend genuinely can't support an operation, **raise `NotImplementedError` with a helpful message** rather than returning a fake result. For example, an HTTP-only "replay" adapter can't `click()`; tell the consumer that:

```python
async def click(self, css_selector: str, *, timeout_ms: int = 5_000) -> None:
    raise NotImplementedError(
        "ReplayBrowserAdapter is read-only — it serves recorded HTML and "
        "can't dispatch input events. Use a live adapter for action steps."
    )
```

Consumers that need a fallback can `try/except NotImplementedError` and degrade gracefully (e.g. the selector grounder doesn't need clicks — only `goto` + `content` + `locator_*`).

### 3. Honour the units

- **Timeouts are milliseconds.** Match Playwright. Convert to seconds inside the method if your backend uses seconds.
- **Selectors are CSS.** Build XPath / role / text locators *on top of* the adapter, not inside it. The adapter shouldn't gain new selector dialects unless we expand the ABC.
- **JS expression and `arg` follow Playwright's rules**: an arrow-function-as-string with `arg` as the first parameter is acceptable, as is a bare expression. `CdpBrowserAdapter.evaluate()` shows the pattern.

### 4. Be honest about what your backend can't do

The `CdpBrowserAdapter` uses JS-level event dispatch for `click` / `fill` / `press` rather than synthesising real mouse/keyboard events via `Input.dispatch*`. This is documented in the source (look for "JS dispatch") and is good enough for most cases but won't trigger hardware-only listeners. If your backend has a similar limitation, surface it in a class-level docstring section called "Known limitations".

### 5. No leaky abstractions

A `BrowserAdapter` consumer must not need to inspect the adapter's type to know what it's doing. If your adapter has knobs that can't be expressed through the ABC, those knobs belong on **the wrapped handle** (which the *caller* configured before instantiating the adapter), not on the adapter itself. Common smell: adding `def get_underlying_page(self) -> PlaywrightPage` so consumers can do backend-specific things. Don't — extend the ABC instead, or refactor the consumer.

## Implementation tips

### Test using `pytest-httpserver`, never real URLs

The CI suite ([`tests/ci/browser/test_adapter.py`](../tests/ci/browser/test_adapter.py)) shows the pattern. Set up an `httpserver` fixture, point your adapter at `httpserver.url_for(...)`, and assert on the *adapter contract*, not on the backend internals.

### Share JS snippets across CSS-locator methods

`CdpBrowserAdapter` synthesises all CSS-based queries via `Runtime.evaluate`. Centralise the JS bodies (or generate them with a shared helper) so the snippet shape stays consistent across methods. The `_css_literal` helper (which JSON-encodes a CSS string for safe embedding in JS) is the prototype.

### Don't fight `accessibility_snapshot`

Different backends produce different tree shapes. Return whatever the backend gives you and document that the keys depend on the adapter. Consumers that need cross-adapter normalisation should layer a normalising helper *on top* — don't try to massage the shapes inside the adapter.

### `close()` is usually a no-op

The adapter doesn't own the underlying session/page — the caller built it, the caller tears it down. Only put cleanup logic in `close()` if your adapter *itself* allocated resources (a thread, a queue, a temp file). Always make `close()` idempotent.

## Wiring a new adapter into the agentic-runtime

Outside this repo, the consumer is the `BrowserToolActor` in [`webrobot-etl-chatbot-server/agentic-runtime`](https://github.com/WebRobot-Ltd/webrobot-etl-chatbot-server). Its three backend modes (`camoufox` / `remote` / `local`) build a Playwright page and wrap it. To add a new adapter from outside this repo:

1. Build whatever handle your adapter wraps (session, page, BaaS client) in the actor's `_ensure_browser()`.
2. Instantiate the adapter: `self._adapter = MyXBrowserAdapter(handle)`.
3. Use `self._adapter` throughout the actor's endpoints (`_visit`, `_observe`, `_ground_selectors`, …) — the actor stays adapter-shaped, the rest is wiring.

When the new adapter is general-purpose enough to live in this repo, send a PR adding it next to `CdpBrowserAdapter` and `PlaywrightBrowserAdapter` in [`browser_use/browser/adapter.py`](../browser_use/browser/adapter.py).

## Extending the contract

If your work needs an operation that's not in the ABC (say, `bounding_box(css_selector)` for visual highlighting), add it to **all three places**:

1. Abstract method on `BrowserAdapter` with a clear docstring and Playwright-compatible signature.
2. Implementations in `CdpBrowserAdapter` and `PlaywrightBrowserAdapter`. Don't leave one out — that's how the contract erodes.
3. A test case in `tests/ci/browser/test_adapter.py` exercising the new method against both implementations.

If you can't implement the operation in `CdpBrowserAdapter` because the CDP protocol genuinely lacks it, raise `NotImplementedError` from the CDP side and tag the operation as "Playwright-only" in the ABC docstring. Future work can either fill it in via JS gymnastics or migrate the consumer off the CDP backend.
