"""End-to-end smoke for the firefox-compat path.

What this validates
-------------------

* :class:`FirefoxPlaywrightEngine` actually launches a Firefox / Camoufox
  subprocess via Playwright Python.
* The launched page can be driven through
  :class:`PlaywrightBrowserAdapter` — same contract the
  ``CdpBrowserAdapter`` exposes for Chromium-via-CDP.
* The two BrowserAdapter methods most commonly used by the
  agentic-runtime selector grounder
  (``accessibility_snapshot_all_frames`` + ``viewport_metrics``) work
  against a real Firefox.

How to run
----------

Requirements (install once)::

    pip install -e .[playwright]          # or `pip install playwright`
    playwright install firefox             # downloads Firefox if missing
    # OPTIONAL — Camoufox instead of stock Firefox:
    pip install camoufox[playwright]
    python -m camoufox fetch

Run::

    python examples/browser/firefox_adapter_smoke.py
    # or with Camoufox:
    CAMOUFOX_BIN=$(python -c "import camoufox, os; print(os.path.join(os.path.dirname(camoufox.__file__), 'binaries'))")
    python examples/browser/firefox_adapter_smoke.py --executable "$CAMOUFOX_BIN/camoufox"

Exit code 0 = everything green. Any failure prints a clear diagnostic
on stderr and exits non-zero.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from browser_use.browser.engine import FirefoxPlaywrightEngine


async def main(url: str, executable: str | None, headless: bool) -> int:
	print(f'→ launching Firefox  (headless={headless}, executable={executable or "default"})')
	handle = await FirefoxPlaywrightEngine.launch_with_adapter(
		headless=headless,
		executable_path=executable,
	)
	adapter = handle['adapter']
	process = handle['process']

	try:
		print(f'  pid: {process.pid if process else "—"}')
		print(f'→ goto {url}')
		nav = await adapter.goto(url, wait_until='domcontentloaded', timeout_ms=30_000)
		print(f'  url={nav.get("url")}  status={nav.get("status")}')

		title = await adapter.title()
		print(f'  title="{title}"')

		print('→ adapter.viewport_metrics()')
		vm = await adapter.viewport_metrics()
		print(f'  width={vm["width"]} height={vm["height"]} dpr={vm["device_pixel_ratio"]}')

		print('→ adapter.locator_count("a") (count anchors)')
		anchors = await adapter.locator_count('a')
		print(f'  {anchors} anchors')

		print('→ adapter.accessibility_snapshot_all_frames()')
		ax = await adapter.accessibility_snapshot_all_frames()
		nodes = ax.get('nodes') or []
		synthetic = sum(1 for n in nodes if n.get('_synthetic'))
		print(f'  {len(nodes)} nodes  (synthetic={synthetic} — expected non-zero on Playwright backend)')

		print('✓ all adapter calls succeeded')
		return 0
	except Exception as e:
		print(f'\n✗ FAILED: {e!r}', file=sys.stderr)
		return 1
	finally:
		print('→ teardown')
		await handle['teardown']()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--url', default='https://books.toscrape.com/',
	                    help='URL to navigate (default: books.toscrape.com)')
	parser.add_argument('--executable', default=None,
	                    help='Path to a Firefox / Camoufox binary (default: Playwright bundled Firefox)')
	parser.add_argument('--headed', action='store_true',
	                    help='Show the browser window (default headless)')
	args = parser.parse_args()
	sys.exit(asyncio.run(main(args.url, args.executable, headless=not args.headed)))
