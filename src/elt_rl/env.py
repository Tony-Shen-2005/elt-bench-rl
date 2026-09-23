"""ELT-Bench RL environment for tinker-cookbook.

Episode lifecycle (one rollout)::

    provision namespace ─► start sandbox ─► write workspace (config.yaml with
    this rollout's destination block, data_model.yaml, schemas, docs)
      ─► agent loop: bash / sql tools, execution feedback, until `submit`,
         max_turns, or context overflow
      ─► ELTReward: kill background processes ─► grade warehouse state
         (stage 1 row counts, stage 2 eval SQL vs ground truth) ─► reward
      ─► cleanup: terraform destroy (airbyte), sandbox, drop namespace

The environment is stateful (warehouse + Airbyte + filesystem). All state is
per rollout, so rollouts in a group can run concurrently.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import chz
import pandas as pd
import yaml

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env

from elt_rl.destinations import Destination, Namespace
from elt_rl.grading import (
    GradeReport,
    ModelResult,
    RewardConfig,
    compare_table,
    compute_reward,
    report_metrics,
)
from elt_rl.sandbox import DockerSandbox, LocalSandbox
from elt_rl.stacks import SYSTEM_PROMPT, AirbyteStack, build_instruction
from elt_rl.task import ELTTask
from elt_rl.tools import ELTTools

logger = logging.getLogger(__name__)

SandboxFactory = Callable[[ELTTask], Awaitable[object]]


async def local_sandbox_factory(task: ELTTask) -> LocalSandbox:
    return LocalSandbox()


TF_PLUGIN_CACHE = Path(os.environ.get("ELTRL_TF_PLUGIN_CACHE", Path.home() / ".cache" / "eltrl-tf-plugins"))


async def docker_sandbox_factory(task: ELTTask) -> DockerSandbox:
    return await DockerSandbox.create(tf_plugin_cache=TF_PLUGIN_CACHE)


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

_GT_CACHE: dict[Path, pd.DataFrame] = {}


def _load_gt(path: Path) -> pd.DataFrame:
    if path not in _GT_CACHE:
        _GT_CACHE[path] = pd.read_csv(path, dtype=str)
    return _GT_CACHE[path]


async def grade(task: ELTTask, destination: Destination, ns: Namespace, *,
                gate_on_stage1: bool = True) -> GradeReport:
    counts = await destination.table_row_counts(ns)
    tables = {t: (counts.get(destination.normalize(t)), exp) for t, exp in task.expected_rows.items()}
    report = GradeReport(tables=tables, models={}, n_models=len(task.models))
    if gate_on_stage1 and not report.stage1_pass:
        return report

    async def one(m) -> tuple[str, ModelResult]:
        if destination.normalize(m.name) not in counts:
            return m.name, ModelResult(exists=False)
        try:
            df = await destination.query(ns, destination.rewrite_eval_sql(ns, m.eval_sql), as_agent=False)
        except Exception as e:
            return m.name, ModelResult(exists=True, error=str(e)[:500])
        return m.name, compare_table(_load_gt(m.gt_csv), df, list(m.sort_keys))

    # Models are independent: grade them concurrently.
    report.models = dict(await asyncio.gather(*(one(m) for m in task.models)))
    return report


@dataclass
class ELTReward:
    """RewardFn: grades the warehouse once, at episode end."""

    task: ELTTask
    destination: Destination
    ns: Namespace
    sandbox: object
    tools: ELTTools
    reward_config: RewardConfig
    transform_only: bool = False
    airbyte: AirbyteStack | None = None
    require_airbyte_provenance: bool = False
    harvest_snapshot: bool = False
    log_dir: Path | None = None
    last_report: GradeReport | None = field(default=None, init=False)

    async def _violations(self) -> list[str]:
        out = []
        if self.require_airbyte_provenance and not self.transform_only:
            # Stage 1 only checks row counts; make sure the rows came through
            # Airbyte (and not, e.g., psql | COPY straight into the warehouse).
            for t in self.task.expected_rows:
                cols = {c.lower() for c in await self.destination.table_columns(self.ns, t)}
                if cols and "_airbyte_raw_id" not in cols:
                    out.append(f"raw table {t} lacks Airbyte metadata columns")
            if self.airbyte is not None:
                try:
                    n = await asyncio.to_thread(self.airbyte.succeeded_syncs_into, self.ns.database)
                    if n == 0:
                        out.append("no succeeded Airbyte sync into this rollout's database")
                except Exception as e:  # audit is best effort; do not punish infra errors
                    logger.warning("Airbyte audit failed: %s", e)
        return out

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        try:
            # Nothing the agent left running may change the warehouse after submission.
            await self.sandbox.kill_background()
            report = await grade(self.task, self.destination, self.ns,
                                 gate_on_stage1=not self.transform_only)
            report.violations = await self._violations()
        except Exception as e:
            logger.exception("grading failed for %s", self.ns.database)
            return 0.0, {"grading_error": 1.0}
        self.last_report = report
        reward = compute_reward(report, self.reward_config, skip_el=self.transform_only)
        metrics = report_metrics(report) | {
            "reward": reward,
            "submitted": float(self.tools.submitted),
            "n_bash": float(self.tools.n_bash),
            "n_sql": float(self.tools.n_sql),
            "grading_error": 0.0,
        }
        if self.harvest_snapshot and report.stage1_pass and not self.transform_only:
            if not await self.destination.has_snapshot(self.task):
                await self.destination.save_snapshot(self.task, self.ns)
                logger.info("saved post-EL snapshot for %s", self.task.name)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"{self.ns.database}.json").write_text(
                json.dumps({"reward": reward, "report": report.to_dict()}, indent=2, default=str))
        return reward, metrics


# --------------------------------------------------------------------------
# Env construction
# --------------------------------------------------------------------------


def workspace_config(task: ELTTask, destination: Destination, ns: Namespace,
                     airbyte: AirbyteStack | None) -> dict:
    config = json.loads(json.dumps(task.config))
    if task.el_stack == "airbyte":
        if airbyte is None:
            raise ValueError("Airbyte tasks need an AirbyteStack")
        config = airbyte.inject(config)
    config.update(destination.agent_config(ns))
    return config


async def write_workspace(sandbox, task: ELTTask, config: dict) -> None:
    ws = sandbox.workspace_path
    await sandbox.write_file(f"{ws}/config.yaml", yaml.safe_dump(config, sort_keys=False))
    await sandbox.write_file(f"{ws}/data_model.yaml", task.data_model_yaml)
    for rel, src in task.workspace_files.items():
        await sandbox.write_file(f"{ws}/{rel}", Path(src).read_bytes())
    await sandbox.run_command(f"mkdir -p {ws}/elt")


class ELTEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        task: ELTTask,
        destination: Destination,
        model_name: str,
        group_size: int,
        renderer_name: str | None = None,
        sandbox_factory: SandboxFactory | None = None,
        airbyte: AirbyteStack | None = None,
        reward_config: RewardConfig = RewardConfig(),
        transform_only: bool = False,
        sync_delay_seconds: int = 0,
        max_turns: int = 60,
        command_timeout: int = 600,
        max_trajectory_tokens: int = 64 * 1024,
        max_generation_tokens: int | None = 4096,
        context_overflow_reward: float = 0.0,
        require_airbyte_provenance: bool = False,
        harvest_snapshot: bool = True,
        log_dir: str | None = None,
        renderer: Renderer | None = None,
    ):
        self.task = task
        self.destination = destination
        self.model_name = model_name
        self.group_size = group_size
        self.renderer_name = renderer_name
        self.sandbox_factory = sandbox_factory or (
            local_sandbox_factory if task.el_stack == "local" else docker_sandbox_factory)
        self.airbyte = airbyte
        self.reward_config = reward_config
        self.transform_only = transform_only
        self.sync_delay_seconds = sync_delay_seconds
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self.context_overflow_reward = context_overflow_reward
        self.require_airbyte_provenance = require_airbyte_provenance
        self.harvest_snapshot = harvest_snapshot
        self.log_dir = Path(log_dir) if log_dir else None
        self._renderer = renderer  # tests inject one; otherwise built lazily
        self._live: list[tuple[object, Namespace]] = []
        self.rewards: list[ELTReward] = []

    def _get_renderer(self) -> Renderer:
        if self._renderer is None:
            tok = tokenizer_utils.get_tokenizer(self.model_name)
            name = self.renderer_name or model_info.get_recommended_renderer_name(self.model_name)
            self._renderer = get_renderer(name, tok)
        return self._renderer

    async def _make_one(self, renderer: Renderer) -> Env:
        rollout_id = uuid.uuid4().hex[:8]
        ns = await self.destination.provision(self.task, rollout_id, from_snapshot=self.transform_only)
        sandbox = await self.sandbox_factory(self.task)
        self._live.append((sandbox, ns))
        await write_workspace(sandbox, self.task, workspace_config(self.task, self.destination, ns, self.airbyte))

        tools = ELTTools(sandbox, self.destination, ns, command_timeout=self.command_timeout)
        reward = ELTReward(
            task=self.task, destination=self.destination, ns=ns, sandbox=sandbox, tools=tools,
            reward_config=self.reward_config, transform_only=self.transform_only,
            airbyte=self.airbyte, require_airbyte_provenance=self.require_airbyte_provenance,
            harvest_snapshot=self.harvest_snapshot, log_dir=self.log_dir,
        )
        self.rewards.append(reward)
        instruction = build_instruction(
            self.task.el_stack, sandbox.workspace_path, self.destination.prompt_notes(ns),
            sync_delay_seconds=self.sync_delay_seconds, transform_only=self.transform_only,
        )
        messages = renderer.create_conversation_prefix_with_tools(
            tools=[t.to_spec() for t in tools.all()], system_prompt=SYSTEM_PROMPT,
        ) + [{"role": "user", "content": instruction}]
        return build_agent_tool_env(
            renderer=renderer, tools=tools.all(), initial_messages=messages, reward_fn=reward,
            max_turns=self.max_turns, max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
            context_overflow_reward=self.context_overflow_reward,
            # A malformed tool call is feedback, not the end of a 60-turn episode.
            failed_parse_reward=0.0, terminate_on_parse_error=False,
        )

    async def make_envs(self) -> Sequence[Env]:
        renderer = self._get_renderer()
        return list(await asyncio.gather(*(self._make_one(renderer) for _ in range(self.group_size))))

    async def cleanup(self) -> None:
        async def one(sandbox, ns):
            try:
                if self.task.el_stack == "airbyte":
                    # Remove this rollout's Airbyte sources/destinations/connections.
                    await sandbox.run_command("cd /workspace/elt && terraform destroy -auto-approve",
                                              timeout=300)
            except Exception as e:
                logger.warning("terraform destroy failed: %s", e)
            for fn in (sandbox.cleanup, lambda: self.destination.teardown(ns)):
                try:
                    await fn()
                except Exception as e:
                    logger.warning("cleanup failed: %s", e)

        await asyncio.gather(*(one(s, n) for s, n in self._live))
        self._live.clear()

    def logging_tags(self) -> list[str]:
        return ["elt_bench", self.task.name, self.destination.name]

    def fresh(self) -> "ELTEnvGroupBuilder":
        """Copy with its own live-resource list, so one task can appear in a
        batch (or in overlapping async batches) more than once."""
        b = copy.copy(self)
        b._live, b.rewards = [], []
        return b


class ELTDataset(RLDataset):
    def __init__(self, builders: list[ELTEnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        n = len(self.builders)
        # Wrap around so training can run more steps than there are tasks.
        return [self.builders[(index * self.batch_size + i) % n].fresh() for i in range(self.batch_size)]

    def __len__(self) -> int:
        return max(1, (len(self.builders) + self.batch_size - 1) // self.batch_size)


@chz.chz
class ELTDatasetBuilder(RLDatasetBuilder):
    tasks: list[ELTTask]
    destination: Destination
    batch_size: int
    group_size: int
    model_name: str
    renderer_name: str | None = None
    sandbox_factory: SandboxFactory | None = None
    airbyte: AirbyteStack | None = None
    reward_config: RewardConfig = RewardConfig()
    transform_only: bool = False
    sync_delay_seconds: int = 0
    max_turns: int = 60
    command_timeout: int = 600
    max_trajectory_tokens: int = 64 * 1024
    max_generation_tokens: int | None = 4096
    require_airbyte_provenance: bool = False
    log_dir: str | None = None
    eval_tasks: list[ELTTask] | None = None

    def _builders(self, tasks: list[ELTTask], group_size: int) -> list[ELTEnvGroupBuilder]:
        return [
            ELTEnvGroupBuilder(
                task=t, destination=self.destination, model_name=self.model_name,
                group_size=group_size, renderer_name=self.renderer_name,
                sandbox_factory=self.sandbox_factory, airbyte=self.airbyte,
                reward_config=self.reward_config, transform_only=self.transform_only,
                sync_delay_seconds=self.sync_delay_seconds, max_turns=self.max_turns,
                command_timeout=self.command_timeout, max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
                require_airbyte_provenance=self.require_airbyte_provenance, log_dir=self.log_dir,
            )
            for t in tasks
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        train = ELTDataset(self._builders(self.tasks, self.group_size), self.batch_size)
        test = (ELTDataset(self._builders(self.eval_tasks, 1), self.batch_size)
                if self.eval_tasks else None)
        return train, test
