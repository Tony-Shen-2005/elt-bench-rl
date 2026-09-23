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

Defaults are `el_weight` 0.2 and `column_credit` 0.5.

The benchmark's own verdict on a task is one bit. `staged` exists to make that bit trainable
without changing what it means, and four decisions follow from that.

**Dense, because GRPO needs within-group variance.** The advantage is measured against the group
mean, so a group scoring all zeros contributes no gradient. Under `binary` that is most early
groups, and an episode that missed one column type looks like one that never ran `terraform init`.

**The completion term is what keeps partial credit honest.** Both endpoints of `column_credit` fail:
at 0, the official all-or-nothing rule gives no signal between nothing and done; at 1, pure linear
credit makes farming the easy columns of every model beat finishing one. In between, the linear part
pays for progress and the `1[f = 1]` part reserves a bonus for a model that is actually finished.

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
| Start the pipeline in the background, submit at once | Background processes killed before grading (tested) |
| Raw tables with the right row counts, garbage content | Capped at the stage 1 share (tested) |
| Load by hand, bypassing Airbyte | `require_airbyte_provenance`: metadata columns on every raw table, plus a succeeded sync job into this rollout's database, audited through the Airbyte API |
| Read the ground truth | Never enters the sandbox |
| Write into another rollout's namespace | Per-rollout database and scoped credential |

Two gaps remain. Nothing verifies the models came from `dbt run`, so loading correctly and then
hand-writing the final tables still scores 1.0. And `_airbyte_raw_id` can be forged, which is why
the binding check is the sync-job audit and not the column check.

## Training efficiency

The cost is wall-clock time, not tokens: an episode waits on `terraform apply` and on sync jobs,
which a better policy does not speed up.

* **Concurrency.** Each rollout gets its own database and container, so a group of rollouts costs
  its slowest member rather than the sum, and the rollouts stay independent samples.
* **Transformation-only episodes.** The first rollout to pass stage 1 saves a post-EL snapshot; with
  `transform_only`, later episodes clone it (zero-copy on Snowflake) and start at stage 2. This
  removes the Airbyte wait entirely and concentrates gradient where most of the reward is.
* **Group composition.** Partial credit makes zero-variance groups rare;
  `remove_constant_reward_groups` drops them, and the asynchronous off-policy path keeps sampling
  while an optimizer step runs, which matters when one rollout takes minutes.
