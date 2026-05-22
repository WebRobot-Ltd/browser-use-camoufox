"""Phase-1 of the firefox-compat porting effort.

These tests lock the data-model surface for multi-engine support:

  - The default ``browser_type`` stays ``CHROMIUM`` (back-compat — every
    existing user of BrowserProfile keeps working untouched).
  - ``BrowserType.FIREFOX`` is accepted and round-trips through pydantic.
  - A profile that mixes ``browser_type=FIREFOX`` with a Chromium-only
    ``channel`` is rejected at validation time, not silently coerced.

Phase-2 will exercise the launch path; this file only checks the data
model so it's safe to ship without the Firefox launcher actually wired.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from browser_use.browser.profile import (
	BROWSERUSE_DEFAULT_BROWSER_TYPE,
	BrowserChannel,
	BrowserProfile,
	BrowserType,
)


def test_default_browser_type_is_chromium() -> None:
	"""Existing profiles default to CHROMIUM — guards back-compat."""
	profile = BrowserProfile()
	assert profile.browser_type == BrowserType.CHROMIUM
	assert BROWSERUSE_DEFAULT_BROWSER_TYPE == BrowserType.CHROMIUM


def test_browser_type_firefox_accepted() -> None:
	profile = BrowserProfile(browser_type=BrowserType.FIREFOX)
	assert profile.browser_type == BrowserType.FIREFOX
	# String form must round-trip identically (Enum's __str__ via str inheritance).
	assert BrowserProfile(browser_type='firefox').browser_type == BrowserType.FIREFOX


def test_browser_type_chromium_keeps_channel() -> None:
	"""Chromium engine + a Chromium channel must be a happy path."""
	profile = BrowserProfile(
		browser_type=BrowserType.CHROMIUM,
		channel=BrowserChannel.CHROME,
	)
	assert profile.channel == BrowserChannel.CHROME


def test_browser_type_firefox_rejects_chromium_channel() -> None:
	"""Channel is Chromium-only — pairing it with FIREFOX is a config bug."""
	with pytest.raises(ValidationError) as exc_info:
		BrowserProfile(
			browser_type=BrowserType.FIREFOX,
			channel=BrowserChannel.CHROME,
		)
	# The error message should call out the conflict (not just "type error").
	assert 'channel' in str(exc_info.value).lower()
	assert 'firefox' in str(exc_info.value).lower()


def test_browser_type_firefox_with_executable_path_ok() -> None:
	"""Firefox variants are selected via executable_path (e.g. Camoufox binary)."""
	profile = BrowserProfile(
		browser_type=BrowserType.FIREFOX,
		executable_path='/usr/local/bin/camoufox',
	)
	assert profile.browser_type == BrowserType.FIREFOX
	assert str(profile.executable_path) == '/usr/local/bin/camoufox'
