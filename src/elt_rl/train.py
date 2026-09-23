"""RL training on ELT tasks with Tinker.

Local, credential-free task (only TINKER_API_KEY needed)::

    uv run python -m elt_rl.train tasks=local max_steps=1

Official ELT-Bench tasks on Snowflake (see README for setup)::

    uv run python -m elt_rl.train tasks=eltbench task_names=amplitude \\
        destination=snowflake eltbench_repo=../ELT-Bench \\
        gt_dir=../ELT-Bench/ground_truth/gt_snowflake max_steps=1
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from elt_rl.destinations import get_destination
from elt_rl.env import ELTDatasetBuilder
from elt_rl.grading import RewardConfig
from elt_rl.stacks import AirbyteStack
from elt_rl.task import load_eltbench_tasks, load_local_task

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]


@chz.chz
class CLIConfig:
    # Tasks
    tasks: Literal["local", "eltbench"] = "local"
    local_task_dir: str = str(REPO / "tests" / "fixtures" / "tiny_shop")
    eltbench_repo: str | None = None
    gt_dir: str | None = None
    task_names: str | None = None  # comma-separated subset
    destination: Literal["duckdb", "snowflake"] = "duckdb"
    duckdb_root: str = "/tmp/elt_rl/warehouse"
    transform_only: bool = False
    require_airbyte_provenance: bool = True
    sync_delay_seconds: int = 0  # ELT-Bench uses 0 for Snowflake, 60 for Databricks

    # Reward
    reward_mode: Literal["binary", "model_fraction", "staged"] = "staged"
    el_weight: float = 0.2
    column_credit: float = 0.5

    # Model / sampling
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer_name: str | None = None
    lora_rank: int = 32
    max_tokens: int = 4096  # per turn
    temperature: float = 1.0
    max_turns: int = 60
    max_trajectory_tokens: int = 32 * 1024
    command_timeout: int = 600

    # RL
    group_size: int = 4
    groups_per_batch: int = 2
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    max_steps: int | None = None
    max_steps_off_policy: int | None = None
    remove_constant_reward_groups: bool = False

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    eval_every: int = 0
    save_every: int = 20
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_dataset_builder(c: CLIConfig) -> ELTDatasetBuilder:
    airbyte = None
    if c.tasks == "local":
        tasks = [load_local_task(c.local_task_dir)]
    else:
        if not (c.eltbench_repo and c.gt_dir):
            raise ValueError("tasks=eltbench needs eltbench_repo and gt_dir")
        names = c.task_names.split(",") if c.task_names else None
        tasks = load_eltbench_tasks(c.eltbench_repo, c.gt_dir, names)
        airbyte = AirbyteStack.from_env()
    if c.destination == "duckdb":
        if c.tasks != "local":
            raise ValueError("Official tasks need a warehouse Airbyte can write to (snowflake)")
        dest = get_destination("duckdb", root=c.duckdb_root)
    else:
        dest = get_destination(c.destination)
    run = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ELTDatasetBuilder(
        tasks=tasks, destination=dest, batch_size=c.groups_per_batch, group_size=c.group_size,
        model_name=c.model_name, renderer_name=c.renderer_name, airbyte=airbyte,
        reward_config=RewardConfig(mode=c.reward_mode, el_weight=c.el_weight, column_credit=c.column_credit),
        transform_only=c.transform_only, sync_delay_seconds=c.sync_delay_seconds,
        max_turns=c.max_turns, command_timeout=c.command_timeout,
        max_trajectory_tokens=c.max_trajectory_tokens, max_generation_tokens=c.max_tokens,
        require_airbyte_provenance=c.require_airbyte_provenance and c.tasks == "eltbench",
        log_dir=f"{c.log_path or '/tmp/elt_rl'}/grades/{run}",
    )


async def cli_main(c: CLIConfig) -> None:
    model_tag = c.model_name.replace("/", "-")
    run_name = f"elt-{c.tasks}-{model_tag}-g{c.group_size}-b{c.groups_per_batch}-{datetime.now():%Y%m%d-%H%M}"
    log_path = c.log_path or f"/tmp/elt_rl/runs/{run_name}"
    c = chz.replace(c, log_path=log_path)
    config = Config(
        learning_rate=c.learning_rate,
        dataset_builder=build_dataset_builder(c),
        model_name=c.model_name,
        recipe_name="elt_bench_rl",
        renderer_name=c.renderer_name or model_info.get_recommended_renderer_name(c.model_name),
        lora_rank=c.lora_rank,
        max_tokens=c.max_tokens,
        temperature=c.temperature,
        kl_penalty_coef=c.kl_penalty_coef,
        log_path=log_path,
        wandb_project=c.wandb_project,
        wandb_name=run_name,
        eval_every=c.eval_every,
        save_every=c.save_every,
        max_steps=c.max_steps,
        remove_constant_reward_groups=c.remove_constant_reward_groups,
        async_config=AsyncConfig(max_steps_off_policy=c.max_steps_off_policy,
                                 groups_per_batch=c.groups_per_batch)
        if c.max_steps_off_policy is not None else None,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=c.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
