"""ELT task specification and loaders.

An ``ELTTask`` has two halves that must never mix:

* the **agent-visible spec**: ``config.yaml`` (sources + destination block),
  ``data_model.yaml``, source schemas, and any stack-specific docs. These are
  copied into the sandbox workspace.
* the **grader-only spec**: expected raw-table row counts, eval SQL, sort keys
  and ground-truth CSVs. These stay on the host and are never written into the
  sandbox (see "reward hacking" in docs/DESIGN.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

ELStack = Literal["airbyte", "local"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    eval_sql: str  # logical SQL, e.g. "select * from address.states order by abbreviation"
    gt_csv: Path
    sort_keys: tuple[str, ...] = ()

    def load_gt(self) -> pd.DataFrame:
        return pd.read_csv(self.gt_csv, dtype=str, keep_default_na=True)


@dataclass(frozen=True)
class ELTTask:
    name: str  # logical schema name used by eval SQL, e.g. "address"
    el_stack: ELStack
    config: dict  # parsed config.yaml WITHOUT the destination block
    data_model_yaml: str
    # Agent-visible files: relative path in workspace -> source path on host
    workspace_files: dict[str, Path] = field(default_factory=dict)
    # Grader-only
    expected_rows: dict[str, int] = field(default_factory=dict)
    models: tuple[ModelSpec, ...] = ()

    @property
    def model_names(self) -> list[str]:
        return [m.name for m in self.models]


# --------------------------------------------------------------------------
# Official ELT-Bench tasks
# --------------------------------------------------------------------------

DESTINATION_KEYS = ("snowflake", "databricks", "redshift")


def load_eltbench_tasks(
    repo: str | Path,
    gt_dir: str | Path,
    names: list[str] | None = None,
) -> list[ELTTask]:
    """Load official tasks from an ELT-Bench checkout.

    Args:
        repo: ELT-Bench repository root (github.com/uiuc-kang-lab/ELT-Bench).
        gt_dir: ground-truth folder for the chosen warehouse, e.g.
            ``ground_truth/gt_snowflake`` from ``hf download tttjjj/elt_bench``.
        names: optional subset of task names.
    """
    repo = Path(repo)
    gt_dir = Path(gt_dir)
    # Destination blocks are stripped and re-added by the Destination, so any
    # warehouse folder works as the source of task definitions.
    bench = repo / "elt-bench" / "snowflake"
    table_json = json.loads((repo / "evaluation" / "table.json").read_text())
    sort_keys = json.loads((repo / "evaluation" / "sort_key.json").read_text())
    docs = sorted((repo / "documentation").glob("*.md"))

    tasks = []
    for task_dir in sorted(p for p in bench.iterdir() if p.is_dir()):
        name = task_dir.name
        if names and name not in names:
            continue
        config = yaml.safe_load((task_dir / "config.yaml").read_text())
        for k in DESTINATION_KEYS:
            config.pop(k, None)
        files: dict[str, Path] = {}
        for csv in sorted((repo / "elt-bench" / "schemas" / name).glob("*.csv")):
            files[f"schemas/{csv.name}"] = csv
        for doc in docs:
            files[f"documentation/{doc.name}"] = doc
        files["check_job_status.py"] = repo / "setup" / "check_job_status.py"
        files["elt/main.tf"] = repo / "setup" / "main.tf"

        models = []
        for sql in sorted((repo / "evaluation" / "sql" / name).glob("*.sql")):
            gt = gt_dir / name / f"{sql.stem}.csv"
            models.append(
                ModelSpec(
                    name=sql.stem,
                    eval_sql=sql.read_text(),
                    gt_csv=gt,
                    sort_keys=tuple(sort_keys.get(name, {}).get(sql.stem, [])),
                )
            )
        tasks.append(
            ELTTask(
                name=name,
                el_stack="airbyte",
                config=config,
                data_model_yaml=(task_dir / "data_model.yaml").read_text(),
                workspace_files=files,
                expected_rows=dict(table_json.get(name, {})),
                models=tuple(models),
            )
        )
    if names:
        missing = set(names) - {t.name for t in tasks}
        if missing:
            raise ValueError(f"Unknown ELT-Bench tasks: {sorted(missing)}")
    return tasks


# --------------------------------------------------------------------------
# Local (credential-free) tasks
# --------------------------------------------------------------------------


def load_local_task(task_dir: str | Path) -> ELTTask:
    """Load a self-contained task in the local-stack layout::

        <task>/
          config.yaml          sources (paths relative to workspace) and task name
          data_model.yaml
          sources/...          raw source files, copied into the workspace
          grader/expected_rows.json
          grader/sql/<model>.sql
          grader/gt/<model>.csv
          grader/sort_key.json
    """
    task_dir = Path(task_dir)
    config = yaml.safe_load((task_dir / "config.yaml").read_text())
    name = config.pop("task")
    files = {
        str(p.relative_to(task_dir)): p
        for p in sorted((task_dir / "sources").rglob("*"))
        if p.is_file()
    }
    grader = task_dir / "grader"
    sort_keys = json.loads((grader / "sort_key.json").read_text())
    models = tuple(
        ModelSpec(
            name=sql.stem,
            eval_sql=sql.read_text(),
            gt_csv=grader / "gt" / f"{sql.stem}.csv",
            sort_keys=tuple(sort_keys.get(sql.stem, [])),
        )
        for sql in sorted((grader / "sql").glob("*.sql"))
    )
    return ELTTask(
        name=name,
        el_stack="local",
        config=config,
        data_model_yaml=(task_dir / "data_model.yaml").read_text(),
        workspace_files=files,
        expected_rows=json.loads((grader / "expected_rows.json").read_text()),
        models=models,
    )
