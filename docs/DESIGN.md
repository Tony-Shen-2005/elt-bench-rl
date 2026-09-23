# Design

This document follows the five questions in the task description: what the task is, how the
reward is designed, how the environment is implemented, how reward hacking is prevented, and
how training is kept affordable.

## 1. The task

ELT-Bench gives an agent a specification and a set of live sources, and asks for a working
pipeline. One task, for example `amplitude`, consists of:

* `config.yaml`: the sources to connect (Postgres, MongoDB, S3, a REST API, flat files), the
  tables to sync from each, and the destination warehouse block.
* `documentation/`: Airbyte provider documentation whose field names do not always match the
  names in `config.yaml`, so values have to be inferred rather than copied.
* `data_model.yaml`: the data models to build, with column descriptions.
* `schemas/`: the source table schemas.

A rollout has to do two things in order.

**Stage 1, extract and load.** Declare Airbyte sources, destinations and connections as
Terraform resources, `terraform apply`, read the connection IDs out of the state file, trigger
one sync job per connection through the Airbyte API, and poll until every job has finished.
The result is a set of raw tables in the warehouse.

**Stage 2, transform.** Write one dbt model per entry in `data_model.yaml`, run them, and
check the resulting tables.

Then the policy calls `submit`, and the warehouse is graded as it stands at that moment.

Two properties shape everything below. The task is **long horizon**: a successful episode is
dozens of tool calls, most of them not directly rewarded. And it is **stateful**: the warehouse,
the Airbyte instance and the filesystem all carry state across turns, so rollouts that share
any of them are not independent samples.

## 2. Reward design

### Two stages, execution based

Grading never looks at the transcript, only at the warehouse.

* **Stage 1** checks that every expected raw table exists with the expected row count, the same
  check as ELT-Bench's `eva_stage1.py`.
* **Stage 2** runs the benchmark's evaluation SQL for each data model and compares the result
  against the ground-truth CSV, column by column, after sorting both sides by the table's key.

The comparator in `grading.py` reproduces `check_corretness` and `sort_by_keys` from
`evaluation/eva_stage2.py` (numeric coercion where possible, relative tolerance 1e-2, NaN
treated as matching NaN). `tests/test_grading.py` imports the official module when a checkout
is available and asserts that both implementations agree, so a rollout that scores 1.0 here
also passes the official evaluator.

### The reward function

Three modes, all derived from the same report:

```
binary          1 if stage 1 passes and every model matches, else 0
model_fraction  fraction of models that match, gated on stage 1
staged          el_weight * stage1_fraction + (1 - el_weight) * stage1_pass * T
```

`staged` is the default, with `el_weight = 0.2`. `T` is the mean model score, where a model
scores 1.0 if it matches exactly and `column_credit` (default 0.5) times its fraction of
matching columns otherwise.

Three decisions are worth stating.

**Why dense at all.** With `binary`, nearly every early rollout scores 0. In GRPO the
advantage is computed within a group, so a group whose rewards are all equal produces no
gradient. A 60-turn episode that ends in a wrong column type would be indistinguishable from
one that never ran `terraform init`, and both would be wasted compute. `staged` keeps the
signal execution derived while making early progress visible: loading three of five source
tables, or getting four of six columns of a model right, moves the reward.

**Why stage 2 is gated on stage 1.** Without the gate, the shortest path to 0.8 is to skip the
pipeline entirely: the agent has shell access to the sources, so it can compute the final
tables directly and insert them. That scores well while doing none of the work the benchmark
is about. The gate also matches the official evaluator, which only evaluates stage 2 for
schemas that stage 1 marked successful, and it imposes a natural curriculum: the EL share is
reachable first, and the larger transformation share only opens once the data is actually
loaded.

**Why the reward is computed once, at the end.** Per-turn rewards would have to be inferred
from tool output, which the policy writes and can therefore shape. Warehouse state at
submission is the one signal the policy can only influence by doing the work. The cost is
sparsity, which is what `staged` is for.

Episodes that end by running out of turns or context receive the same grading as any other,
scoring whatever the warehouse deserves at that point, except for context overflow which
scores 0.

## 3. Implementation

```
provision namespace -> start sandbox -> write workspace
  -> agent loop: bash / sql / submit, with execution feedback
  -> kill background processes -> grade warehouse -> reward
  -> terraform destroy, stop sandbox, drop namespace
```

**Destination abstraction.** `Destination` owns everything warehouse-specific: provisioning an
isolated namespace, the connection block the agent sees, SQL execution, row counts, identifier
case folding, and rewriting the benchmark's logical `task.table` references into the rollout's
own namespace. The environment, the tools and the grader only talk to this interface. Adding
BigQuery or Redshift means writing one subclass. Two are implemented: Snowflake, the official
warehouse, laid out as ELT-Bench does with tables in `<database>.AIRBYTE_SCHEMA`; and DuckDB,
a file-backed warehouse used by the credential-free local task.

**Isolation.** A GRPO group runs several rollouts of the same task concurrently, and they must
not see each other's tables. Every rollout gets its own database, `ELTRL_<TASK>_<ROLLOUT>` on
Snowflake or its own file on DuckDB, and its own container. Eval SQL is rewritten per rollout,
so the grader reads the namespace that rollout wrote.

**Sandbox.** One Docker container per rollout, from ELT-Bench's `elt-swe` image, attached to the
benchmark's network so it can reach the sources and Airbyte. Only the rollout's workspace is
mounted. A local subprocess sandbox exists for the credential-free tests, with no isolation, and
is never used for training.

**What the agent can and cannot see.** `ELTTask` is deliberately split in two. The agent-visible
half, `config.yaml`, `data_model.yaml`, source schemas and documentation, is written into the
sandbox. The grader-only half, expected row counts, evaluation SQL, sort keys and ground-truth
CSVs, stays on the host and is never written into the sandbox or into a prompt.

**Credentials.** The Snowflake destination holds two. The admin credential creates and drops
databases and performs grading, and is never shown to the agent. The agent credential is the
benchmark's `AIRBYTE_USER` and `AIRBYTE_ROLE`, written into the agent's `config.yaml` and used
by Airbyte and dbt. The `sql` tool executes with the agent credential, so what the agent can
inspect is exactly what its pipeline can reach.

**Tools.** Three: `bash` in the sandbox, `sql` against the rollout's namespace with the agent
credential and a 50-row cap, and `submit`, which ends the episode. Errors are returned as
tool output rather than raised, because an error message is the feedback the policy is
supposed to learn from. A malformed tool call is also returned as feedback and does not end
the episode.

**Cleanup.** After grading, `terraform destroy` removes the rollout's Airbyte resources, the
container is stopped and the namespace is dropped. Without this, Airbyte accumulates one set
of sources, destinations and connections per rollout, and the warehouse accumulates databases.

## 4. Reward hacking

The official evaluator checks results only. Stage 1 counts rows and does not look at content or
provenance, stage 2 compares the final tables and does not check that dbt produced them, and
the instruction to use Airbyte and dbt exists only in the prompt. For a single evaluation run of
a general agent that is reasonable. Under RL it is not: a shortcut that scores well will be
reinforced whether or not it was intended, so anything that is only asked for in the prompt has
to become a check in the environment.

| Shortcut | Handling |
| --- | --- |
| Submit immediately | Scores 0. Covered by a scripted test. |
| Start the real pipeline in the background, then submit at once and hope it finishes during grading | Background processes are killed before grading, so the warehouse is empty. Covered by a scripted test. |
| Create raw tables with the right row counts and garbage content | Capped at the stage 1 share, because the models built on top cannot match. Covered by a scripted test. |
| Load the raw tables by hand, for example `psql` piped into the warehouse, bypassing Airbyte | `require_airbyte_provenance`, on by default for the official tasks, requires Airbyte metadata columns on every raw table and at least one succeeded sync job into this rollout's database, queried from the Airbyte API by the grader. |
| Skip extraction and loading and write the final tables directly from the sources | Stage 2 is gated on stage 1, so this scores 0. |
| Read the ground truth | Ground truth, expected row counts and eval SQL never enter the sandbox, and the container mounts only the workspace. |
| Write into another rollout's namespace | Each rollout owns its own database and container, and the agent credential is scoped to it. |

Two gaps remain, both worth naming rather than hiding.

**Provenance is checked for loading, not for transformation.** Nothing verifies that the data
models were produced by `dbt run`. An agent that loads correctly through Airbyte and then
hand-writes the final tables with SQL still scores 1.0. The natural fix is to require dbt's own
artifacts, `target/run_results.json`, to list every model, and to compare the model names and
timestamps against the tables found in the warehouse.

**The metadata column check is weak on its own.** `_airbyte_raw_id` can be added by hand. The
sync-job audit against the Airbyte API is the check that actually binds, and the column check
is a cheap precondition. If the API audit fails for infrastructure reasons it is logged and
skipped rather than punished, which is the right call for a flaky local deployment but does
mean the weaker check can be the only one in force.

## 5. Training efficiency

The cost of this task is wall-clock time, not tokens. A full episode waits on `terraform apply`
and on Airbyte sync jobs, which take minutes and are not made faster by a better policy.

* **Concurrency.** All rollouts in a group, and all groups in a batch, run concurrently. This is
  what the per-rollout namespace and container buy: the wall-clock cost of a group is close to
  the cost of its slowest member rather than their sum.
* **Transformation-only episodes.** The first rollout that passes stage 1 saves a post-EL
  snapshot of its namespace, with raw tables only. With `transform_only`, later episodes clone
  that snapshot and start at stage 2 (a zero-copy database clone on Snowflake, a file copy
  on DuckDB). This removes the Airbyte wait entirely and concentrates
  gradient on the part of the task where most of the reward lives. It is the natural curriculum:
  train the transformation stage against snapshots, then train end to end.
* **Terraform plugin cache.** Containers share a provider cache directory, so `terraform init`
  does not re-download the Airbyte provider for every rollout.
* **Caps.** 60 turns, 4096 generated tokens per turn, 32k trajectory tokens, 600s per command.
  A runaway episode is bounded, and a trajectory that would exceed the context is ended rather
  than silently truncated.
* **Group composition.** Partial credit exists partly for throughput: it makes all-equal groups,
  which contribute no gradient, rare. `remove_constant_reward_groups` can drop them outright, and
  the asynchronous off-policy path in tinker-cookbook lets sampling continue while an optimizer
  step runs, which matters when a single rollout takes minutes.

## 6. Verification status and limitations

Passing: the credential-free integration tests, including the scripted oracle and the four
scripted failure and hacking policies, and comparator parity with the official evaluator.

Not yet run: a credentialed rollout against a real Snowflake and Airbyte deployment, and a
Tinker optimization step. Both are needed before any efficiency number in this document can be
replaced with a measurement.

## 7. Future work

The ground truth is treated here as correct. ELT-Bench-Verified (arXiv 2603.29399) reports that
roughly 1.2% of ground-truth columns are ambiguous, admitting more than one defensible answer,
and the original benchmark does not remove them. Under evaluation that is a small constant
error. Under RL it is not neutral: those columns hand out reward that no policy can earn
reliably, which is the regime studied in *Noisy Data is Destructive to RLVR* (arXiv 2603.16140).
A concrete next step is to grade against the verified subset, and to compare training under the
original and verified ground truth, which would measure how much of this environment's learning
signal is noise.
