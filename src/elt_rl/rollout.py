"""Run and grade rollouts without training (for debugging and the credentialed check).

    # policy = a Tinker base model (needs TINKER_API_KEY)
    uv run python -m elt_rl.rollout tasks=eltbench task_names=amplitude destination=snowflake \\
        eltbench_repo=../ELT-Bench gt_dir=../ELT-Bench/ground_truth/gt_snowflake

Prints each rollout's reward and grade report, and saves the transcript.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import chz
import tinker

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_single_rollout

from elt_rl.train import CLIConfig, build_dataset_builder


@chz.chz
class RolloutConfig(CLIConfig):
    n_rollouts: int = 1
    checkpoint: str | None = None  # tinker://... sampler path; default: base model
    out_dir: str = "/tmp/elt_rl/rollouts"


async def run(c: RolloutConfig) -> None:
    c = chz.replace(c, group_size=c.n_rollouts, groups_per_batch=1, log_path=c.out_dir)
    train, _ = await build_dataset_builder(c)()
    builder = train.get_batch(0)[0]
    service = tinker.ServiceClient()
    sc = (service.create_sampling_client(model_path=c.checkpoint, base_model=c.model_name)
          if c.checkpoint else service.create_sampling_client(base_model=c.model_name))
    policy = TinkerTokenCompleter(sc, max_tokens=c.max_tokens, temperature=c.temperature)
    tok = tokenizer_utils.get_tokenizer(c.model_name)
    out = Path(c.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    envs = await builder.make_envs()
    try:
        trajs = await asyncio.gather(*(do_single_rollout(policy, e) for e in envs))
        for i, (traj, rew) in enumerate(zip(trajs, builder.rewards)):
            final = traj.transitions[-1]
            report = rew.last_report.to_dict() if rew.last_report else None
            print(f"\n=== rollout {i} ({rew.ns.database}) turns={len(traj.transitions)} "
                  f"reward={final.reward:.3f}")
            print(json.dumps(report, indent=2, default=str))
            transcript = [tok.decode(t.ac.tokens) for t in traj.transitions]
            (out / f"{rew.ns.database}.json").write_text(json.dumps(
                {"reward": final.reward, "metrics": final.metrics, "report": report,
                 "assistant_turns": transcript}, indent=2, default=str))
    finally:
        await builder.cleanup()
    print(f"\ntranscripts: {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(chz.entrypoint(RolloutConfig)))
