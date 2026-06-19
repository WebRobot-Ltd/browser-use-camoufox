"""Claude Agent SDK backend for browser_use.

Routes ``ainvoke()`` through the **Claude Agent SDK** (the Claude Code runtime) instead of
the raw Anthropic Messages API, so browser_use's agentic loop runs on a Claude *subscription*
(``CLAUDE_CODE_OAUTH_TOKEN``) the same way an interactive Claude Code session does — with the
subscription's allowance + built-in retries — rather than on a metered API key that hits the
raw-API per-model rate limits. The full browser_use loop (DOM serialization, element indexing,
action loop) is unchanged; only the "brain" backend differs.

Why this exists
---------------
browser_use's bundled LLM clients (``ChatAnthropic`` etc.) call ``api.anthropic.com`` directly.
With a Max-plan subscription token that surface is rate-limited per model (e.g. Sonnet) while
the Claude Code path keeps working. This shim lets a self-hosted browser_use agent reuse the
same subscription/Claude-Code path the rest of the stack uses.

Auth
----
Pass ``oauth_token=...`` or set ``CLAUDE_CODE_OAUTH_TOKEN`` in the environment (mint one with
``claude setup-token`` — it does not invalidate existing tokens). A pay-as-you-go ``api_key``
(or ``ANTHROPIC_API_KEY``) is also accepted. **Never hard-code the credential.**

Limitations
-----------
Text-only today — run the agent with ``Agent(..., use_vision=False)`` so no screenshots are
sent. Vision passthrough via the SDK content blocks is a future refinement. Each ``ainvoke``
is a one-shot ``query`` (stateless per step), which matches browser_use's per-step contract.
"""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel

from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar("T", bound=BaseModel)


def _message_to_text(message: BaseMessage) -> str:
	"""Flatten a browser_use message into ``ROLE: text``. Image parts are dropped
	(use ``use_vision=False``); only text parts are forwarded."""
	role = getattr(message, "role", "user") or "user"
	content = getattr(message, "content", "")
	if isinstance(content, list):
		parts: list[str] = []
		for part in content:
			text = getattr(part, "text", None)
			if text is None and isinstance(part, dict):
				text = part.get("text")
			if text:
				parts.append(text)
		content = "\n".join(parts)
	return f"{str(role).upper()}: {content}"


class ChatClaudeAgentSdk:
	"""A browser_use ``BaseChatModel`` whose calls go through the Claude Agent SDK.

	Implements the structural ``BaseChatModel`` protocol (``model``, ``provider``,
	``name``, ``model_name``, ``ainvoke``). Drop it into ``Agent(llm=...)``.
	"""

	_verified_api_keys: bool = True

	def __init__(
		self,
		model: str = "claude-haiku-4-5",
		*,
		oauth_token: str | None = None,
		api_key: str | None = None,
		permission_mode: str = "bypassPermissions",
		max_turns: int = 1,
	) -> None:
		self.model = model
		# Resolved at runtime from args or env — never stored in source.
		self._oauth_token = oauth_token or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
		self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
		self._permission_mode = permission_mode
		self._max_turns = max_turns

	@property
	def provider(self) -> str:
		return "claude-agent-sdk"

	@property
	def name(self) -> str:
		return self.model

	@property
	def model_name(self) -> str:
		return self.model

	def _options(self):
		from claude_agent_sdk import ClaudeAgentOptions

		env: dict[str, str] = {}
		if self._oauth_token:
			env["CLAUDE_CODE_OAUTH_TOKEN"] = self._oauth_token
		elif self._api_key:
			env["ANTHROPIC_API_KEY"] = self._api_key
		return ClaudeAgentOptions(
			model=self.model,
			max_turns=self._max_turns,
			permission_mode=self._permission_mode,
			env=env or None,
		)

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[Any]:
		from claude_agent_sdk import query

		prompt = "\n\n".join(_message_to_text(m) for m in messages)
		if output_format is not None:
			prompt += (
				"\n\nReturn ONLY one JSON object matching this schema "
				"(no prose, no markdown fences):\n" + json.dumps(output_format.model_json_schema())
			)

		text = ""
		async for msg in query(prompt=prompt, options=self._options()):
			if type(msg).__name__ == "AssistantMessage":
				for block in getattr(msg, "content", []):
					if type(block).__name__ == "TextBlock":
						text += block.text

		if output_format is None:
			return ChatInvokeCompletion(completion=text, usage=None)

		start, end = text.find("{"), text.rfind("}")
		raw = text[start : end + 1] if start != -1 and end != -1 else text
		return ChatInvokeCompletion(completion=output_format.model_validate_json(raw), usage=None)
