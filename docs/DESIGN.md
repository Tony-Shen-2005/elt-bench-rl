# Design

Repository: https://github.com/Tony-Shen-2005/elt-bench-rl

Two properties of the task drive everything here. It is **long horizon**: a successful episode is
dozens of tool calls, almost none individually rewarded. And it is **stateful**: the warehouse,
the Airbyte deployment and the filesystem persist across turns, so rollouts sharing any of them
are not independent samples, and a reset costs minutes rather than nothing.

## Reward

Grading turns the warehouse into a report: for each raw table, rows found against rows expected;
for each model, which ground-truth columns match after key-aware sorting. Three functions turn
that report into a scalar.

```
binary          1 iff stage 1 passes and every model matches
model_fraction  fraction of models matching, gated on stage 1
staged          el_weight * stage1_fraction + (1 - el_weight) * stage1_pass * T   (default)
```

`T` is the mean per-model score. Writing `f` for the fraction of ground-truth columns a model
matches, and scoring 0 unless its row count is right, that score is

```
column_credit * f + (1 - column_credit) * 1[f = 1]
```

an interpolation between the official all-or-nothing rule (`column_credit = 0`) and pure linear
credit (`column_credit = 1`). Defaults are `el_weight` 0.2 and `column_credit` 0.5.

**Dense, because GRPO needs within-group variance.** Under `binary` nearly every early rollout
scores 0, the advantage is zero across the group, and an episode that failed on one column type is
indistinguishable from one that never ran `terraform init`. Every point in `staged` is still
execution derived.

**A premium on finishing.** A model with five of six columns right is not 83% of a pipeline, it is
unusable, and the benchmark scores it 0. Under pure linear credit the last column is worth no more
than the first, so the best strategy is to farm the cheap columns of every model and finish none.
The completion term keeps one finished model ahead of two half-finished ones.

**Stage 2 gated on stage 1.** Ungated, the cheapest path to the transformation share is to skip the
pipeline: the agent can read the sources through `bash` and write the final tables directly. The
gate also imposes a curriculum, since the larger share only opens once the data is really loaded.

**Outcome, not process.** Per-turn rewards would have to be read out of tool output, which the
policy writes and can therefore shape. Warehouse state at submission is the one signal the policy
can only move by doing the work.

**Full credit means the official evaluator passes.** The comparator reproduces `check_corretness`
and `sort_by_keys`, and a test asserts agreement with `eva_stage2.py` table by table, so the reward
cannot drift from the benchmark it claims to optimize. `task_success` and `srdt_fraction` are logged
every step, so a run trained on `staged` is still reported in the benchmark's own terms.

**Integrity violations zero the reward** rather than reduce it, so cheating plus good work never
outscores honest work.

## Environment

* **Everything warehouse-specific behind `Destination`:** namespace provisioning, the agent's
  connection block, SQL execution, row counts, identifier case, eval-SQL rewriting. Snowflake and
  DuckDB are implemented; a new warehouse is one subclass and nothing else changes. DuckDB is what
  makes a credential-free integration test possible.
* **One database and one container per rollout.** A group runs concurrently against shared
  infrastructure, so isolation is what keeps rollouts independent samples; eval SQL is rewritten
  into the rollout's own namespace.
* **A task is split in two at the type level.** Agent-visible files go into the workspace;
  expected row counts, eval SQL, sort keys and ground truth stay on the host and cannot be reached
  from the container.
* **Two credentials.** Admin provisions and grades and is never shown. The agent credential is the
  benchmark's `AIRBYTE_USER`, also used by the `sql` tool, so the agent can inspect exactly what
  its pipeline can reach.
* **Errors are observations.** Command failures, SQL errors and malformed tool calls come back as
  tool results. Ending a forty-turn episode over a parse error discards everything learned in it.
* **Background processes are killed before grading, and resources are destroyed after.** Otherwise
  a rollout can keep writing after submission, and Airbyte accumulates one resource set per episode.

## Reward hacking

Under RL, anything required only by the prompt will eventually be skipped, so prompt-level
requirements became environment-level checks.

| Shortcut | Handling |
| --- | --- |
| Submit immediately | Scores 0 (tested) |
| Start the pipeline in the background, submit at once | Background processes killed before grading (tested) |
| Raw tables with the right row counts, garbage content | Capped at the stage 1 share (tested) |
| Load by hand, bypassing Airbyte | `require_airbyte_provenance`: metadata columns on every raw table, plus a succeeded sync job into this rollout's database, audited through the Airbyte API |
| Skip EL, write the final tables from the sources | Gated, scores 0 |
| Read the ground truth | Never enters the sandbox |
| Write into another rollout's namespace | Per-rollout database and scoped credential |

Two gaps are open. Nothing verifies the models came from `dbt run`, so loading correctly and then
hand-writing the final tables still scores 1.0; requiring `target/run_results.json` to account for
every model would close it. And `_airbyte_raw_id` can be forged, so the binding check is the
sync-job audit, with the column check as a cheap precondition.

## Training efficiency

The cost is wall-clock time, not tokens: an episode waits on `terraform apply` and on sync jobs,
which a better policy does not speed up.

* **Concurrency** is what per-rollout isolation buys: a group costs its slowest member, not the sum.
* **Transformation-only episodes.** The first rollout to pass stage 1 saves a post-EL snapshot; with
  `transform_only`, later episodes clone it (zero-copy on Snowflake) and start at stage 2. This
  removes the Airbyte wait entirely and concentrates gradient where most of the reward is.
* **Shared Terraform plugin cache**, so `terraform init` does not re-download the provider per rollout.
* **Caps:** 60 turns, 4096 generated tokens per turn, 32k trajectory tokens, 600s per command.
* **Group composition.** Partial credit makes zero-variance groups rare;
  `remove_constant_reward_groups` drops them, and the asynchronous off-policy path keeps sampling
  while an optimizer step runs, which matters when one rollout takes minutes.

## Future work

Ground truth is treated here as correct. ELT-Bench-Verified (arXiv 2603.29399) reports that about
1.2% of ground-truth columns admit more than one defensible answer and are not removed from the
original benchmark. Under evaluation that is a small constant error; under RL it hands out reward
no policy can earn reliably, the regime studied in *Noisy Data is Destructive to RLVR* (arXiv
2603.16140). The next step is to grade against the verified subset and compare training under the
original and verified ground truth, measuring how much of this environment's learning signal is noise.
