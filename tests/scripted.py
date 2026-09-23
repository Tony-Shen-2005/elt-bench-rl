"""Scripted policies for credential-free rollouts.

tinker's rollout loop is ``policy(tokens) -> action tokens -> renderer.parse_response``.
We keep the real loop, env, tools, sandbox, warehouse and grader, and replace
only the model: each env gets a script of assistant messages. The policy emits
two tokens ``[script_id, turn]`` and the renderer decodes them back into that
script's message, so several scripts can share one renderer inside a group.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import tinker

from tinker_cookbook.completers import TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers.base import Message, ParseTermination, ToolCall

_ids = itertools.count()


def call(name: str, **arguments: Any) -> Message:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [ToolCall(id=f"call_{next(_ids)}",
                                function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments)))],
    }


class ScriptBook:
    def __init__(self) -> None:
        self.scripts: list[list[Message]] = []

    def add(self, script: list[Message]) -> "ScriptedPolicy":
        self.scripts.append(script)
        return ScriptedPolicy(len(self.scripts) - 1)


class ScriptedPolicy(TokenCompleter):
    def __init__(self, script_id: int):
        self.script_id = script_id
        self.turn = 0

    async def __call__(self, model_input: tinker.ModelInput, stop: Any, *,
                       max_tokens: int | None = None) -> TokensWithLogprobs:
        toks = [self.script_id, self.turn]
        self.turn += 1
        return TokensWithLogprobs(tokens=toks, maybe_logprobs=[0.0, 0.0], stop_reason="stop")


class GroupPolicy(TokenCompleter):
    """One script for every env in a group; tracks turns per prompt lineage."""

    def __init__(self, script_id: int, n_msgs_initial: int):
        self.script_id = script_id
        self.n0 = n_msgs_initial

    async def __call__(self, model_input: tinker.ModelInput, stop: Any, *,
                       max_tokens: int | None = None) -> TokensWithLogprobs:
        # The fake prompt is one token per message; each turn adds 2 (assistant + tool).
        turn = (model_input.length - self.n0) // 2
        return TokensWithLogprobs(tokens=[self.script_id, turn], maybe_logprobs=[0.0, 0.0],
                                  stop_reason="stop")


class ScriptRenderer:
    """Minimal renderer: one token per message; parses ``[script_id, turn]``."""

    def __init__(self, book: ScriptBook):
        self.book = book

    def get_stop_sequences(self) -> list[str]:
        return ["<stop>"]

    def create_conversation_prefix_with_tools(self, tools: list, system_prompt: str = "") -> list[Message]:
        return [{"role": "system", "content": system_prompt + "\n" + json.dumps(tools)}]

    def build_generation_prompt(self, messages: list[Message], **kwargs: Any) -> tinker.ModelInput:
        return tinker.ModelInput.from_ints([0] * len(messages))

    def parse_response(self, action: list[int]):
        script_id, turn = action
        return self.book.scripts[script_id][turn], ParseTermination.STOP_SEQUENCE
