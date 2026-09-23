# Design

Repository: https://github.com/Tony-Shen-2005/elt-bench-rl

## 1. The task

An ELT-Bench task gives the agent a specification (`config.yaml`: sources and the tables to sync;
`data_model.yaml`: the models to build; source schemas; Airbyte documentation whose field names
deliberately do not match the config) and a set of live sources. A rollout declares Airbyte
sources, destinations and connections in Terraform, applies them, triggers one sync job per
connection and polls until they finish, then writes dbt models, runs them, and submits. Its tools
are `bash` in a sandbox, `sql` against the warehouse with the pipeline's own credentials, and
`submit`. An episode ends on `submit`, on 60 turns, or on context overflow.

Two properties drive every decision below. The task is **long horizon**: a successful episode is
dozens of tool calls, almost none of them individually rewarded. And it is **stateful**: the
warehouse, the Airbyte deployment and the filesystem persist across turns, so rollouts that share
any of them are not independent samples.

## 2. Reward

Grading reads the warehouse, never the transcript. Stage 1 checks that every expected raw table
exists with the expected row count. Stage 2 runs the benchmark's evaluation SQL per model and
compares against ground truth column by column. The comparator reproduces `check_corretness` and
`sort_by_keys` from `evaluation/eva_stage2.py`, and a test asserts agreement with the official
module, so a rollout scoring 1.0 here also passes the official evaluator.

```
binary          1 iff stage 1 passes and every model matches
model_fraction  fraction of models matching, gated on stage 1
staged          el_weight * stage1_fraction + (1 - el_weight) * stage1_pass * T   (default)
```

`T` is the mean model score: 1.0 for an exact match, otherwise `column_credit` times the fraction
of matching columns. Defaults are `el_weight` 0.2 and `column_credit` 0.5.

**Why dense.** Under `binary` nearly every early rollout scores 0. GRPO computes the advantage
within a group, so an all-equal group yields no gradient: an episode that failed on one column
type would be indistinguishable from one that never ran `terraform init`, and both would be
wasted compute. `staged` stays execution derived while making partial progress visible.

**Why stage 2 is gated on stage 1.** Ungated, the cheapest path to the transformation share is to
skip the pipeline: the agent can read the sources through `bash` and write the final tables
directly. The gate also matches the official evaluator, which only evaluates stage 2 for schemas
that stage 1 marked successful, and it imposes a curriculum, since the transformation share only
opens once the data is really loaded.

**Outcome, not process.** Per-turn rewards would have to be read out of tool output, which the
policy writes and can therefore shape. Warehouse state at submission is the one signal the policy
can only move by doing the work. The cost is sparsity, which is what `staged` addresses.

## 3. Implementation

```
provision namespace -> start sandbox -> write workspace
  -> agent loop: bash / sql / submit, with execution feedback
  -> kill background processes -> grade warehouse -> reward
  -> terraform destroy, stop sandbox, drop namespace
```

* **Destination abstraction.** Namespace provisioning, the agent's connection block, SQL
  execution, row counts, identifier case and eval-SQL rewriting sit behind one interface, and the
  environment, tools and grader talk only to it. Snowflake and DuckDB are implemented; adding a
  warehouse is one subclass.
* **Isolation.** A group runs concurrently, so each rollout owns a database
  (`ELTRL_<TASK>_<ROLLOUT>`) and a container, and eval SQL is rewritten into that namespace.
* **Sandbox.** One container per rollout from ELT-Bench's `elt-swe` image, on the benchmark's
  network, mounting only that rollout's workspace.
* **Two halves of a task.** Agent-visible files go into the workspace. Expected row counts, eval
  SQL, sort keys and ground truth stay on the host.
* **Two credentials.** The admin credential provisions and grades and is never shown. The agent
  credential is the benchmark's `AIRBYTE_USER`, used by Airbyte, dbt and the `sql` tool, so the
  agent can inspect exactly what its pipeline can reach.
* **Feedback, not failure.** Command errors, SQL errors and malformed tool calls come back as
  tool results; they are what the policy is meant to learn from.
* **Cleanup.** `terraform destroy`, container stop and namespace drop after grading, otherwise
  Airbyte accumulates one resource set per rollout.

## 4. Reward hacking

The official evaluator checks results only: stage 1 counts rows without looking at content or
provenance, stage 2 does not check that dbt built the tables, and "use Airbyte and dbt" lives only
in the prompt. For one evaluation run of a general agent that is fine. Under RL it is not, because
a shortcut that scores gets reinforced, so prompt-level requirements have to become
environment-level checks.

| Shortcut | Handling |
| --- | --- |
| Submit immediately | Scores 0 (tested) |
| Start the pipeline in the background, submit at once | Background processes are killed before grading (tested) |
| Raw tables with the right row counts, garbage content | Capped at the stage 1 share, the models cannot match (tested) |
| Load by hand, bypassing Airbyte | `require_airbyte_provenance`: Airbyte metadata columns on every raw table, plus a succeeded sync job into this rollout's database, audited through the Airbyte API |
| Skip EL, write the final tables from the sources | Stage 2 is gated on stage 1, scores 0 |
| Read the ground truth | It never enters the sandbox; only the workspace is mounted |
| Write into another rollout's namespace | Per-rollout database and scoped credential |

Two gaps are open. Nothing verifies that the data models came from `dbt run`, so loading correctly
and then hand-writing the final tables still scores 1.0; requiring dbt's `target/run_results.json`
to account for every model would close it. And `_airbyte_raw_id` can be forged, so the binding
check is the sync-job audit, with the column check as a cheap precondition.

## 5. Training efficiency

The cost here is wall-clock time, not tokens: an episode waits on `terraform apply` and on Airbyte
sync jobs, which a better policy does not speed up.

* **Concurrency.** Rollouts in a group and groups in a batch run concurrently, which is what
  per-rollout namespaces and containers buy: a group costs its slowest member, not the sum.
* **Transformation-only episodes.** The first rollout to pass stage 1 saves a post-EL snapshot
  (raw tables only). With `transform_only`, later episodes clone it (zero-copy on Snowflake, a file
  copy on DuckDB) and start at stage 2, removing the Airbyte wait and concentrating gradient where
  most of the reward is. It is also the natural curriculum.
* **Terraform plugin cache.** Shared across containers, so `terraform init` does not re-download
  the provider for every rollout.
* **Caps.** 60 turns, 4096 generated tokens per turn, 32k trajectory tokens, 600s per command.
* **Group composition.** Partial credit makes all-equal groups rare, `remove_constant_reward_groups`
  drops them outright, and the asynchronous off-policy path keeps sampling while an optimizer step
  runs, which matters when one rollout takes minutes.

## 6. Future work

Ground truth is treated here as correct. ELT-Bench-Verified (arXiv 2603.29399) reports that about
1.2% of ground-truth columns admit more than one defensible answer and are not removed from the
original benchmark. Under evaluation that is a small constant error; under RL it hands out reward
that no policy can earn reliably, the regime studied in *Noisy Data is Destructive to RLVR* (arXiv
2603.16140). The next step is to grade against the verified subset and compare training under the
original and verified ground truth, measuring how much of this environment's learning signal is
noise.
