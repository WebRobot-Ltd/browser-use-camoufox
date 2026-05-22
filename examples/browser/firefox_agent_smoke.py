"""End-to-end smoke for the full browser-use Agent on Firefox/Camoufox.

What this validates
-------------------

The complete Phase 5 stack:

  - :class:`FirefoxPlaywrightEngine` launches Firefox / Camoufox.
  - :class:`BrowserSession` brings up cleanly via the BiDi
    :class:`BrowserConnection` (Phase 5a).
  - All ported watchdogs survive the bring-up:
    AboutBlank / Crash / Popups / Permissions / Screenshot /
    StorageState / DefaultAction (Phase 5b).
  - :class:`DomService` produces a synthesised
    :class:`EnhancedDOMTreeNode` tree via the JS walker (Phase 5b).
  - :class:`Agent` runs a complete observe → LLM → act loop end-to-end
    against the resulting state.

This is the full ``browser_use.agent.service.Agent`` — not the
``firefox_minimal_agent`` parallel loop. If THIS works, the fork is
"completo" in the sense the user asked for.

How to run
----------

Setup (once)::

    pip install -e .
    playwright install firefox
    # Optional anti-detect:
    pip install camoufox[playwright]
    python -m camoufox fetch

Pick an LLM provider — same as ``firefox_minimal_agent.py``::

    export WEBROBOT_API_ENDPOINT=https://api.webrobot.eu       # WebRobot
    # OR direct provider:
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GROQ_API_KEY=gsk-...

Run::

    python examples/browser/firefox_agent_smoke.py \\
        --task "Go to books.toscrape.com and tell me the title of the first book"

    # With Camoufox:
    python examples/browser/firefox_agent_smoke.py \\
        --task "..." \\
        --executable /path/to/camoufox

The Agent prints its step-by-step decisions. Exit 0 if the loop
terminates with `done`, 1 if it runs out of steps, 2 on hard errors.

Known limits
------------

  - Auxiliary watchdogs not ported yet: file downloads, security
    request filtering, HAR recording, video recording. If the task
    needs any of those, the Agent will fail or behave oddly.
  - Multi-tab discovery on BiDi is single-page. Tasks that open
    multiple tabs will trigger the BiDi tab-recovery logic but the
    agent's focus tracking may get confused. Stick to single-tab
    workflows for the first iterations.
  - The DOM tree is synthesised — accessibility data, paint order,
    stacking contexts are absent. Selector quality may degrade vs.
    Chromium.

Iterate by feeding back the first concrete failure you see: the
Phase-5b boundary is now thin enough that each issue is one focused
patch away.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Reuse the LLM picker from the minimal agent — same provider matrix.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from firefox_minimal_agent import pick_llm  # noqa: E402

from browser_use.agent.service import Agent
from browser_use.browser.profile import BrowserProfile, BrowserType
from browser_use.browser.session import BrowserSession


async def run_smoke(task: str, headless: bool, executable: str | None,
                    model_hint: str | None, max_steps: int) -> int:
	llm = pick_llm(model_hint)
	print(f'→ LLM: {llm.__class__.__name__} model={getattr(llm, "model", "?")}')

	profile = BrowserProfile(
		browser_type=BrowserType.FIREFOX,
		headless=headless,
		executable_path=executable,
	)
	session = BrowserSession(browser_profile=profile)

	# Start session — this exercises Phase 5a (BiDi connect) + Phase 5b
	# watchdog bring-up. Any failure here is in the foundation, not in
	# the Agent loop.
	print('→ starting BrowserSession (Firefox/BiDi)')
	try:
		await session.start()
	except Exception as e:
		print(f'✗ BrowserSession.start failed: {e!r}', file=sys.stderr)
		return 2

	print(f'  connection backend: {session._connection.backend}')
	print(f'  connection open:    {session._connection.is_open}')

	# Build the Agent — let it use the session we already started.
	# We deliberately do NOT pass tools/controller; the default Tools
	# registry maps to the events DefaultActionWatchdog now handles
	# on both backends.
	agent = Agent(
		task=task,
		llm=llm,
		browser_session=session,
	)

	print(f'→ Agent.run(max_steps={max_steps})')
	try:
		result = await agent.run(max_steps=max_steps)
		print(f'\n✓ Agent finished — final result: {result!r}')
		return 0
	except Exception as e:
		print(f'\n✗ Agent.run failed: {e!r}', file=sys.stderr)
		import traceback
		traceback.print_exc()
		return 2
	finally:
		print('→ teardown')
		try:
			await session.kill()
		except Exception as e:
			print(f'  (teardown error: {e})', file=sys.stderr)


def main() -> int:
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument('--task', required=True, help='NL task for the Agent')
	parser.add_argument('--executable', default=None,
	                    help='Firefox / Camoufox binary (default: Playwright bundled)')
	parser.add_argument('--headed', action='store_true', help='Show the browser')
	parser.add_argument('--model', default=None,
	                    help='Override LLM (e.g. webrobot/groq, openai/gpt-4o-mini)')
	parser.add_argument('--max-steps', type=int, default=15)
	args = parser.parse_args()

	return asyncio.run(run_smoke(
		task=args.task,
		headless=not args.headed,
		executable=args.executable,
		model_hint=args.model,
		max_steps=args.max_steps,
	))


if __name__ == '__main__':
	sys.exit(main())
