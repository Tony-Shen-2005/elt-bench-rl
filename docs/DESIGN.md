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

A task asks for several data models. Each one is scored on its own, and `T` is the mean of those
scores. Writing `f` for the fraction of a model's ground-truth columns that match after key-aware
sorting, that model scores

```
s = column_credit * f + (1 - column_credit) * 1[f = 1]      s = 0 if the row count is wrong
T = mean of s over the models of the task
```

`s` interpolates between the official all-or-nothing rule (`column_credit = 0`) and pure linear
credit (`column_credit = 1`). Defaults are `el_weight` 0.2 and `column_credit` 0.5.

The benchmark's own verdict on a task is one bit. `staged` exists to make that bit trainable
without changing what it means, and four decisions follow from that.

**Dense, because GRPO needs within-group variance.** The advantage is measured against the group
mean, so a group scoring all zeros contributes no gradient. Under `binary` that is most early
groups, and an episode that missed one column type looks like one that never ran `terraform init`.

**The completion term is what keeps partial credit honest.** All-or-nothing credit gives no signal
between nothing and done; pure linear credit makes farming the easy columns of every model beat
finishing one. `column_credit` interpolates, paying for progress but reserving a bonus for a model
that is actually finished.

**Gating stage 2 on stage 1 is both a defence and a curriculum.** Ungated, the cheapest route to the
larger share is to skip the pipeline: read the sources with `bash` and write the final tables by
hand. Gated, the transformation reward opens only once the data is really loaded.

**Full credit is defined by the official evaluator, not by us.** A test asserts the comparator agrees
with `eva_stage2.py` table by table, and `task_success` is logged every step. The shaping changes
where the gradient comes from, never what counts as solved.

## Reward hacking

Under RL, anything required only by the prompt will eventually be skipped, so prompt-level
requirements became environment-level checks. A detected violation zeroes the reward rather than
reducing it, so cheating plus good work never outscores honest work.

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

* **Concurrency.** Each rollout gets its own database and container, so a group of rollouts costs
  its slowest member rather than the sum, and the rollouts stay independent samples.
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
