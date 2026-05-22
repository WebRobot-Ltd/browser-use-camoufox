"""BrowserConnection ABC + factory contract tests.

Phase-5a foundation. These tests check the connection-layer contract
without launching a real browser — pure interface + factory behaviour.
The end-to-end "BidiBrowserConnection can attach to a Playwright Firefox
WS endpoint and stay open" check needs a live browser; that's covered
by ``examples/browser/firefox_adapter_smoke.py`` (run locally; not
gated on CI infra it would require).
"""

from __future__ import annotations

import inspect
import pytest

from browser_use.browser.connection import (
	BidiBrowserConnection,
	BrowserConnection,
	CdpBrowserConnection,
	connection_from_browser_type,
)


# ── ABC contract ─────────────────────────────────────────────────────────────


def test_browser_connection_is_abstract() -> None:
	with pytest.raises(TypeError):
		BrowserConnection()  # type: ignore[abstract]


def test_concrete_connections_implement_every_abstract_method() -> None:
	abstract = {
		name for name, method in BrowserConnection.__dict__.items()
		if getattr(method, '__isabstractmethod__', False)
	}
	# start/stop are async; is_open is a sync property.
	assert 'start' in abstract and 'stop' in abstract and 'is_open' in abstract

	for cls in (CdpBrowserConnection, BidiBrowserConnection):
		assert not cls.__abstractmethods__, (
			f'{cls.__name__} still has abstract methods: {cls.__abstractmethods__}'
		)


def test_backend_identifiers() -> None:
	"""Backend marker is the only way consumers can branch on protocol —
	guard it. ``base`` would mean someone forgot to set it on a subclass."""
	assert CdpBrowserConnection.backend == 'cdp'
	assert BidiBrowserConnection.backend == 'bidi'


# ── Factory ──────────────────────────────────────────────────────────────────


def test_factory_routes_known_types() -> None:
	assert connection_from_browser_type('chromium') is CdpBrowserConnection
	assert connection_from_browser_type('firefox')  is BidiBrowserConnection


def test_factory_rejects_unknown_type() -> None:
	with pytest.raises(ValueError, match='browser_type'):
		connection_from_browser_type('webkit')


# ── BidiBrowserConnection constructor enforcement ────────────────────────────


def test_bidi_connection_requires_browser_or_ws_endpoint() -> None:
	"""You can't build a BidiBrowserConnection with nothing — guards
	the ``None / None`` constructor mistake that would silently produce
	a connection that does nothing on start()."""
	with pytest.raises(ValueError, match='browser.*or.*ws_endpoint'):
		BidiBrowserConnection()


def test_bidi_connection_with_ws_endpoint_only() -> None:
	"""When only ws_endpoint is supplied, owns the playwright runtime
	(it will be created on start). is_open must report False until
	start() runs."""
	conn = BidiBrowserConnection(ws_endpoint='ws://localhost:1234/x')
	assert conn.backend == 'bidi'
	assert conn.is_open is False
	# Accessing the browser before start should raise — preserves the
	# 'don't pretend you're connected when you aren't' invariant.
	with pytest.raises(RuntimeError, match='start'):
		_ = conn.browser


# ── Method signatures stay aligned across impls ──────────────────────────────


def test_signatures_match_abc() -> None:
	"""If an adapter renames a parameter, kwargs calls break silently.
	Lock the signatures (matches the equivalent test on BrowserAdapter).
	Properties decorated with ``@abstractmethod`` are skipped — there's
	no callable signature to align, only the property name."""
	for name, abc_method in BrowserConnection.__dict__.items():
		if not getattr(abc_method, '__isabstractmethod__', False):
			continue
		# Properties wrap a fget callable but inspect.signature on the
		# property object itself raises TypeError. Skip them — name
		# parity is the only contract for property-shaped abstractions.
		if isinstance(abc_method, property):
			for cls in (CdpBrowserConnection, BidiBrowserConnection):
				assert hasattr(cls, name), f'{cls.__name__} missing property {name!r}'
			continue
		abc_sig = inspect.signature(abc_method)
		for cls in (CdpBrowserConnection, BidiBrowserConnection):
			impl_sig = inspect.signature(getattr(cls, name))
			assert list(impl_sig.parameters) == list(abc_sig.parameters), (
				f'{cls.__name__}.{name} param order diverges from ABC: '
				f'{list(impl_sig.parameters)} vs {list(abc_sig.parameters)}'
			)
