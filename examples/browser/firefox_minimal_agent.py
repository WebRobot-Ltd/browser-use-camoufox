"""Minimal LLM-driven agent on Firefox / Camoufox via BrowserAdapter.

What this is
------------

A focused observe → decide → act loop running on the firefox-compat
fork. It uses *only* what we've shipped so far:

  - :class:`FirefoxPlaywrightEngine` to spawn Firefox (or Camoufox via
    ``--executable``).
  - :class:`PlaywrightBrowserAdapter` for every page operation.
  - One of the bundled :mod:`browser_use.llm` chat clients (OpenAI,
    Anthropic, Groq, Google) to make decisions.

What this is NOT
----------------

The full ``Agent`` from :mod:`browser_use.agent.service`. That one
needs Phase-5b/c/d watchdog ports before it runs on Firefox. This
minimal agent skips watchdogs entirely and drives the page through the
adapter, which DOES work end-to-end on Firefox today.

Why ship it
-----------

A real working loop on Firefox/Camoufox lets you collaudo the whole
stack we've built — engine + adapter + connection layer — without
waiting for the watchdog refactors. Once Phase-5 is complete this
example becomes a "minimal alternative" pattern that some callers may
still prefer (smaller dependency surface, no event bus, no agent state
to manage).

Run
---

Install (once)::

    pip install -e .
    playwright install firefox
    # Optional anti-detect:
    pip install camoufox[playwright]
    python -m camoufox fetch

Pick ONE provider — set the matching env var(s)::

    # WebRobot LLM endpoint (canonical for this project — Groq → OpenAI →
    # Anthropic → TogetherAI auto-cascade managed server-side).
    export WEBROBOT_API_ENDPOINT=https://api.webrobot.eu       # base URL
    # The WebRobot demo endpoints don't need a credential; for /api/llm/infer
    # specifically the server resolves credentials from the platform's cloud
    # vault. Setting WEBROBOT_API_KEY only attributes usage for analytics.
    export WEBROBOT_LLM_PROVIDER=groq                          # optional hint

    # OR a direct provider (browser-use's bundled clients):
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export GROQ_API_KEY=gsk-...
    export GOOGLE_API_KEY=...

Provider auto-pick order (when ``--model`` is not given): WebRobot endpoint
first (since it exists in every WebRobot deploy), then OpenAI, Anthropic,
Groq, Google. To force a specific provider use ``--model``, e.g.::

    --model webrobot/groq               # WebRobot endpoint with provider hint
    --model openai/gpt-4o-mini
    --model anthropic/claude-3-5-haiku-latest
    --model groq/llama-3.3-70b-versatile

Run with a goal::

    python examples/browser/firefox_minimal_agent.py \\
        --goal "Go to books.toscrape.com and tell me the title of the first book on the page" \\
        --start-url https://books.toscrape.com/

    # With Camoufox:
    python examples/browser/firefox_minimal_agent.py \\
        --goal "..." \\
        --executable /path/to/camoufox

The agent logs each step (observation summary + LLM decision + action
result). Exit code 0 if the agent reaches ``done``, 1 if it exhausts
the max iterations, 2 on launch / LLM errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

import httpx

from browser_use.browser.adapter import PlaywrightBrowserAdapter
from browser_use.browser.engine import FirefoxPlaywrightEngine
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import AssistantMessage, BaseMessage, SystemMessage, UserMessage
from browser_use.llm.views import ChatInvokeCompletion


# ── Prompt ───────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a precise browser-driving agent. The user gives you a goal; you observe the page and decide ONE action at a time.

Available actions:
  - goto(url)               : navigate to a URL
  - click(css_selector)     : click the first element matching the selector
  - fill(css_selector, value): clear and fill an input
  - done(answer)            : finish, returning your answer to the user

OUTPUT EXACTLY one JSON object per turn — no prose, no markdown fences. Examples:
  {"action": "goto",  "url": "https://example.com"}
  {"action": "click", "selector": "article.product_pod h3 a"}
  {"action": "fill",  "selector": "input[name=q]", "value": "books"}
  {"action": "done",  "answer": "The first book is A Light in the Attic."}

Selector tips:
  - Prefer stable selectors (semantic tags, specific class names) over nth-child.
  - You only see a truncated page; if you don't see what you need, navigate or click further.
  - Be definitive: never output two actions, never explain.
"""


# ── WebRobot LLM endpoint client ─────────────────────────────────────────────


class _WebRobotChat(BaseChatModel):
	"""Thin BaseChatModel client for the WebRobot LLM endpoint.

	Endpoint contract (mirrors what the agentic-runtime BrowserToolActor
	uses)::

	    POST {base}/api/webrobot/api/llm/infer
	    body: {"prompt": str, "systemPrompt": str?, "provider": str?}
	    response: {"result": str, ...}

	Auto-cascades Groq → OpenAI → Anthropic → TogetherAI server-side, so
	the client doesn't need provider-specific config beyond an optional
	``provider`` hint. Credentials live in the server's cloud vault; the
	client only sends ``Authorization`` if WEBROBOT_API_KEY is set
	(strictly for org-level usage attribution, not for auth on the demo
	endpoint).
	"""

	# Public BaseChatModel-required attrs.
	model: str = 'webrobot'

	def __init__(self, *, base_url: str, provider: str | None = None,
	             api_key: str | None = None, timeout_s: float = 60.0) -> None:
		# BaseChatModel is a Protocol-shaped pydantic-compatible parent; we
		# don't call super().__init__() because there's no concrete base
		# state to initialise. Attributes are class-level + instance-level.
		self.base_url = base_url.rstrip('/')
		self.provider_hint = provider
		self.api_key = api_key
		self.timeout_s = timeout_s
		self.model = f'webrobot/{provider}' if provider else 'webrobot'

	@property
	def provider(self) -> str:
		return 'webrobot'

	@property
	def name(self) -> str:
		return self.model

	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[str]:
		# Flatten messages into prompt + systemPrompt. Our endpoint
		# accepts only those two fields; multi-turn history is folded
		# into `prompt` as a transcript so the LLM still sees the
		# conversation shape.
		system_parts: list[str] = []
		convo_parts: list[str] = []
		for m in messages:
			content = _message_content_to_text(m)
			if isinstance(m, SystemMessage):
				system_parts.append(content)
			elif isinstance(m, AssistantMessage):
				convo_parts.append(f'Assistant: {content}')
			else:
				# UserMessage or unknown — treat as user input.
				convo_parts.append(content if isinstance(m, UserMessage) else f'{m.__class__.__name__}: {content}')

		payload: dict[str, Any] = {'prompt': '\n\n'.join(convo_parts)}
		if system_parts:
			payload['systemPrompt'] = '\n\n'.join(system_parts)
		if self.provider_hint:
			payload['provider'] = self.provider_hint

		headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
		if self.api_key:
			headers['Authorization'] = (
				self.api_key if self.api_key.lower().startswith(('bearer ', 'apikey '))
				else f'ApiKey {self.api_key}'
			)

		# NOTE: the path is /api/webrobot/api/llm/infer — Tomcat mounts
		# Jersey at /api/* so the @Path("/webrobot/api/llm") resource
		# lives at /api/webrobot/api/llm. Same convention as every other
		# WebRobot endpoint.
		url = f'{self.base_url}/api/webrobot/api/llm/infer'
		async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s, connect=15.0)) as client:
			r = await client.post(url, json=payload, headers=headers)
			r.raise_for_status()
			data = r.json()

		text = data.get('result') or data.get('text') or ''
		if not text:
			raise RuntimeError(f'WebRobot LLM returned no `result`: {data}')

		# output_format is currently ignored — the WebRobot endpoint
		# doesn't enforce structured output. Callers that need it can
		# json.loads(completion) themselves; the MinimalAgent already
		# does so via _parse_action().
		return ChatInvokeCompletion(completion=text, usage=None)


def _message_content_to_text(m: BaseMessage) -> str:
	"""Flatten a BaseMessage's content into a plain string.

	Browser-use messages carry content as either ``str`` or a list of
	content-part dicts (text / image / refusal). For our endpoint we
	keep only the text parts.
	"""
	content = getattr(m, 'content', None)
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts: list[str] = []
		for p in content:
			# Pydantic content parts have .type + .text fields.
			if isinstance(p, dict):
				if p.get('type') == 'text' and 'text' in p:
					parts.append(p['text'])
			else:
				txt = getattr(p, 'text', None)
				if txt:
					parts.append(txt)
		return '\n'.join(parts)
	return '' if content is None else str(content)


# ── LLM picker ───────────────────────────────────────────────────────────────


def pick_llm(model_hint: str | None) -> Any:
	"""Return a configured Chat* client based on env vars and an optional
	model hint. Lazy-import the providers so missing optional deps don't
	break the script when another provider is selected."""
	if model_hint:
		# Explicit override: parse `provider/model` or `provider:model`.
		# For `webrobot`, the second part is interpreted as the optional
		# provider hint (groq/openai/anthropic/togetherai), since the
		# concept of "model" lives server-side under the WebRobot endpoint.
		m = re.match(r'(?P<provider>[a-z]+)[/:](?P<model>.+)', model_hint)
		if not m:
			# Allow bare `webrobot` (no provider hint).
			if model_hint.strip().lower() == 'webrobot':
				provider, model = 'webrobot', ''
			else:
				raise ValueError(f'--model must look like provider/model, got {model_hint!r}')
		else:
			provider, model = m.group('provider'), m.group('model')
	else:
		# Auto-pick — WebRobot endpoint first when its base URL is set
		# (canonical path for this project). Then individual providers
		# by env var.
		if os.environ.get('WEBROBOT_API_ENDPOINT'):
			provider, model = 'webrobot',  os.environ.get('WEBROBOT_LLM_PROVIDER') or ''
		elif os.environ.get('OPENAI_API_KEY'):
			provider, model = 'openai',    'gpt-4o-mini'
		elif os.environ.get('ANTHROPIC_API_KEY'):
			provider, model = 'anthropic', 'claude-3-5-haiku-latest'
		elif os.environ.get('GROQ_API_KEY'):
			provider, model = 'groq',      'llama-3.3-70b-versatile'
		elif os.environ.get('GOOGLE_API_KEY'):
			provider, model = 'google',    'gemini-2.0-flash'
		else:
			raise RuntimeError(
				'No LLM provider configured. Set WEBROBOT_API_ENDPOINT (recommended), '
				'or export one of OPENAI_API_KEY / ANTHROPIC_API_KEY / GROQ_API_KEY / '
				'GOOGLE_API_KEY, or pass --model.'
			)

	if provider == 'webrobot':
		base = os.environ.get('WEBROBOT_API_ENDPOINT', 'https://api.webrobot.eu')
		return _WebRobotChat(
			base_url=base,
			provider=(model or os.environ.get('WEBROBOT_LLM_PROVIDER') or None),
			api_key=os.environ.get('WEBROBOT_API_KEY'),
		)
	if provider == 'openai':
		from browser_use.llm.openai.chat import ChatOpenAI
		return ChatOpenAI(model=model)
	if provider == 'anthropic':
		from browser_use.llm.anthropic.chat import ChatAnthropic
		return ChatAnthropic(model=model)
	if provider == 'groq':
		from browser_use.llm.groq.chat import ChatGroq
		return ChatGroq(model=model)
	if provider == 'google':
		from browser_use.llm.google.chat import ChatGoogle
		return ChatGoogle(model=model)
	raise ValueError(f'unsupported provider {provider!r}')


# ── Observation ──────────────────────────────────────────────────────────────


async def _observe(adapter: PlaywrightBrowserAdapter) -> dict[str, Any]:
	"""Snapshot the current page for the LLM's user message.

	We keep this terse on purpose — overstuffing the prompt with full
	HTML burns tokens and degrades decisions. The accessibility tree
	captures interactable structure; we add a small HTML excerpt for
	context the a11y tree might miss (text content, attribute values).
	"""
	url = await adapter.url()
	title = await adapter.title()

	html = await adapter.content()
	# Heuristic body excerpt — drop scripts/styles, keep the rest small.
	body_idx = html.find('<body')
	excerpt = html[body_idx : body_idx + 6000] if body_idx >= 0 else html[:6000]

	ax_full = await adapter.accessibility_snapshot_all_frames()
	ax_nodes = ax_full.get('nodes') or []
	# Keep only nodes with a real role/name — drops decorative noise.
	terse = [
		{
			'role':  n.get('role',  {}).get('value'),
			'name':  n.get('name',  {}).get('value'),
			'value': n.get('value', {}).get('value') if n.get('value') else None,
		}
		for n in ax_nodes
		if (n.get('role') or {}).get('value')
		   and (n.get('name') or {}).get('value')
	][:80]  # cap

	return {
		'url':           url,
		'title':         title,
		'accessibility': terse,
		'html_excerpt':  excerpt,
	}


# ── Decision parsing ─────────────────────────────────────────────────────────


_JSON_OBJECT_RE = re.compile(r'\{.*\}', re.DOTALL)


def _parse_action(text: str) -> dict[str, Any]:
	"""Extract the first JSON object from the LLM completion.

	Tolerates models that wrap output in markdown fences despite the
	instruction not to — the regex picks the first ``{...}`` span,
	``json.loads`` confirms it.
	"""
	stripped = text.strip()
	# Quick path: whole completion IS a JSON object.
	try:
		obj = json.loads(stripped)
		if isinstance(obj, dict):
			return obj
	except Exception:
		pass
	# Fallback: pluck the first {...} span.
	m = _JSON_OBJECT_RE.search(stripped)
	if not m:
		raise ValueError(f'no JSON object in LLM completion: {stripped[:200]!r}')
	obj = json.loads(m.group(0))
	if not isinstance(obj, dict):
		raise ValueError(f'expected JSON object, got {type(obj).__name__}')
	return obj


# ── Loop ─────────────────────────────────────────────────────────────────────


async def run_agent(
	*,
	goal: str,
	start_url: str | None,
	headless: bool,
	executable: str | None,
	model_hint: str | None,
	max_iterations: int,
) -> tuple[int, str | None]:
	"""Returns ``(exit_code, final_answer)``."""
	llm = pick_llm(model_hint)
	print(f'→ LLM: {llm.__class__.__name__} model={llm.model}')

	handle = await FirefoxPlaywrightEngine.launch_with_adapter(
		headless=headless, executable_path=executable,
	)
	adapter: PlaywrightBrowserAdapter = handle['adapter']

	try:
		if start_url:
			print(f'→ goto {start_url}')
			await adapter.goto(start_url, wait_until='domcontentloaded')

		history: list[dict[str, Any]] = []
		for turn in range(1, max_iterations + 1):
			obs = await _observe(adapter)
			print(f'\n──── turn {turn}  ({obs["url"]})')
			print(f'  title: {obs["title"]!r}')
			print(f'  ax nodes: {len(obs["accessibility"])}')

			user_msg = (
				f'Goal: {goal}\n\n'
				f'URL: {obs["url"]}\n'
				f'Title: {obs["title"]!r}\n\n'
				f'Accessibility tree (truncated):\n{json.dumps(obs["accessibility"], indent=1)}\n\n'
				f'HTML excerpt (truncated):\n{obs["html_excerpt"]}\n\n'
				f'Action history: {json.dumps(history, indent=1) if history else "(empty)"}\n\n'
				f'Output your next action as a single JSON object.'
			)

			try:
				resp = await llm.ainvoke([
					SystemMessage(content=SYSTEM_PROMPT),
					UserMessage(content=user_msg),
				])
				text = resp.completion if hasattr(resp, 'completion') else str(resp)
			except Exception as e:
				print(f'  ✗ LLM call failed: {e}', file=sys.stderr)
				return 2, None

			try:
				action = _parse_action(text)
			except Exception as e:
				print(f'  ✗ could not parse action: {e}\n  raw: {text[:300]!r}', file=sys.stderr)
				history.append({'turn': turn, 'unparsed': text[:200]})
				continue

			print(f'  → {action}')
			history.append({'turn': turn, **action})

			kind = action.get('action')
			try:
				if kind == 'done':
					answer = action.get('answer') or '(no answer)'
					print(f'\n✓ DONE: {answer}')
					return 0, answer
				elif kind == 'goto':
					url = action.get('url') or ''
					if not url.startswith(('http://', 'https://')):
						raise ValueError(f'goto: invalid URL {url!r}')
					await adapter.goto(url, wait_until='domcontentloaded')
				elif kind == 'click':
					await adapter.click(action['selector'], timeout_ms=5_000)
				elif kind == 'fill':
					await adapter.fill(action['selector'], action.get('value', ''),
					                   timeout_ms=5_000)
				else:
					print(f'  ✗ unknown action: {kind!r}', file=sys.stderr)
			except Exception as e:
				print(f'  ! action failed (will continue): {e}')
				history[-1]['error'] = str(e)[:200]

		print(f'\n⚠ reached max iterations ({max_iterations}) without "done"', file=sys.stderr)
		return 1, None

	finally:
		print('\n→ teardown')
		await handle['teardown']()


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--goal', required=True, help='NL goal for the agent')
	parser.add_argument('--start-url', default=None, help='Initial URL (optional)')
	parser.add_argument('--executable', default=None,
	                    help='Path to a Firefox / Camoufox binary (default: Playwright bundled Firefox)')
	parser.add_argument('--headed', action='store_true', help='Show the browser window')
	parser.add_argument('--model', default=None,
	                    help='Force a model: provider/model (e.g. openai/gpt-4o-mini, anthropic/claude-3-5-haiku-latest, groq/llama-3.3-70b-versatile, google/gemini-2.0-flash)')
	parser.add_argument('--max-iterations', type=int, default=15,
	                    help='Hard cap on the observe→act loop (default: 15)')
	args = parser.parse_args()

	exit_code, _ = asyncio.run(run_agent(
		goal=args.goal,
		start_url=args.start_url,
		headless=not args.headed,
		executable=args.executable,
		model_hint=args.model,
		max_iterations=args.max_iterations,
	))
	return exit_code


if __name__ == '__main__':
	sys.exit(main())
