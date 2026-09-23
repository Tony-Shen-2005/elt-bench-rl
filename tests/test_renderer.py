"""The real chat template renders our tools and parses the model's tool calls.

Downloads the (public) tokenizer from Hugging Face; skipped when offline.
"""

import asyncio
import json
from pathlib import Path

import pytest

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer

from elt_rl.destinations.duckdb import DuckDBDestination
from elt_rl.env import ELTEnvGroupBuilder
from elt_rl.task import load_local_task

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@pytest.fixture(scope="module")
def renderer():
    try:
        tok = tokenizer_utils.get_tokenizer(MODEL)
    except Exception as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")
    return get_renderer(model_info.get_recommended_renderer_name(MODEL), tok), tok


def test_initial_observation_and_tool_call_parsing(renderer, tmp_path):
    r, tok = renderer
    task = load_local_task(Path(__file__).parent / "fixtures" / "tiny_shop")

    async def go():
        b = ELTEnvGroupBuilder(task=task, destination=DuckDBDestination(tmp_path), model_name=MODEL,
                               group_size=1, renderer=r)
        env = (await b.make_envs())[0]
        obs, stop = await env.initial_observation()
        await b.cleanup()
        return obs

    obs = asyncio.run(go())
    prompt = tok.decode(obs.to_ints())
    for name in ("bash", "sql", "submit"):
        assert f'"name": "{name}"' in prompt
    assert "data_model.yaml" in prompt and obs.length < 4000

    text = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call><|im_end|>'
    msg, term = r.parse_response(tok.encode(text, add_special_tokens=False))
    assert term.is_clean
    (tc,) = msg["tool_calls"]
    assert tc.function.name == "bash" and json.loads(tc.function.arguments) == {"command": "ls -la"}
