"""Execution-based grading for ELT rollouts.

Two stages, mirroring ELT-Bench's official evaluator:

* Stage 1 (extract & load): every expected raw table exists in the rollout's
  namespace with exactly the expected row count.
* Stage 2 (transform): every data model, fetched with the benchmark's eval SQL,
  matches the ground-truth CSV column by column.

The comparator below reproduces ``check_corretness`` / ``sort_by_keys`` from
ELT-Bench ``evaluation/eva_stage2.py`` (main @ fcf3129), so a rollout that
scores 1.0 here also passes the official evaluator. ``tests/test_grading.py``
checks parity against an ELT-Bench checkout when one is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Comparator (ELT-Bench evaluation/eva_stage2.py)
# --------------------------------------------------------------------------


def _sort_series_key(col: pd.Series) -> pd.Series:
    num = pd.to_numeric(col, errors="coerce")
    n_non_null = col.notna().sum()
    if n_non_null > 0 and int(num.notna().sum()) == int(n_non_null):
        return num
    return col.astype("string").str.strip().str.lower()


def sort_by_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Sort by the table's unique key, then every other column as tie-breaker."""
    if df.empty:
        return df.reset_index(drop=True)
    lower = {c.lower(): c for c in df.columns}
    resolved: list[str] = []
    for k in keys:
        actual = lower.get(k.lower())
        if actual is not None and actual not in resolved:
            resolved.append(actual)
    ordered = resolved + [c for c in df.columns if c not in resolved]
    aux = pd.DataFrame({c: _sort_series_key(df[c]) for c in ordered})
    order = aux.sort_values(by=ordered, kind="mergesort", na_position="last").index
    return df.loc[order].reset_index(drop=True)


def _to_numeric_if_possible(s: pd.Series) -> pd.Series:
    coerced = pd.to_numeric(s, errors="coerce")
    if coerced.isna().sum() == s.isna().sum():
        return coerced
    return s


def vectors_match(v1: pd.Series, v2: pd.Series, rtol: float = 1e-2, atol: float = 1e-9) -> bool:
    if len(v1) != len(v2):
        return False
    n1 = _to_numeric_if_possible(v1.reset_index(drop=True))
    n2 = _to_numeric_if_possible(v2.reset_index(drop=True))
    if pd.api.types.is_numeric_dtype(n1) and pd.api.types.is_numeric_dtype(n2):
        a = n1.to_numpy(dtype=float)
        b = n2.to_numpy(dtype=float)
        nan_match = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
        return bool(np.all(nan_match | close))
    for a, b in zip(v1, v2):
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b):
            return False
        if isinstance(a, str) and isinstance(b, str):
            if a.strip().lower() != b.strip().lower():
                return False
        elif a != b:
            return False
    return True


def compare_table(df_gt: pd.DataFrame, df: pd.DataFrame, keys: list[str]) -> "ModelResult":
    """Compare a candidate table with ground truth after key-aware sorting.

    Both sides are compared as strings (``dtype=str`` in the official
    evaluator, which round-trips through CSV), so we stringify here too.
    """
    df_gt = sort_by_keys(_as_str(df_gt), keys)
    df = sort_by_keys(_as_str(df), keys)
    cols = {c.lower(): c for c in df.columns}
    matched, unmatched, missed = [], [], []
    for gold in df_gt.columns:
        actual = cols.get(gold.lower())
        if actual is None:
            missed.append(gold)
        elif vectors_match(df_gt[gold], df[actual]):
            matched.append(gold)
        else:
            unmatched.append(gold)
    return ModelResult(
        exists=True,
        row_count_ok=len(df) == len(df_gt),
        match=not unmatched and not missed,
        matched=matched,
        unmatched=unmatched,
        missed=missed,
    )


def _as_str(df: pd.DataFrame) -> pd.DataFrame:
    """Mimic ``df.to_csv(); pd.read_csv(dtype=str)``: values become strings, NULL stays NaN."""
    out = pd.DataFrame(index=range(len(df)))
    for c in df.columns:
        col = df[c].reset_index(drop=True)
        out[str(c)] = col.map(_cell_to_str).astype(object)
    return out


def _cell_to_str(v: object) -> object:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    s = str(v)
    return np.nan if s == "" else s


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@dataclass
class ModelResult:
    exists: bool
    row_count_ok: bool = False
    match: bool = False
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def column_fraction(self) -> float:
        total = len(self.matched) + len(self.unmatched) + len(self.missed)
        return len(self.matched) / total if total else 0.0


@dataclass
class GradeReport:
    # Stage 1: table -> (found rows or None if missing, expected rows)
    tables: dict[str, tuple[int | None, int]]
    # Stage 2: model name -> result (empty when stage 1 failed and gating is on)
    models: dict[str, ModelResult]
    n_models: int
    violations: list[str] = field(default_factory=list)

    @property
    def stage1_fraction(self) -> float:
        if not self.tables:
            return 1.0
        ok = sum(1 for found, exp in self.tables.values() if found == exp)
        return ok / len(self.tables)

    @property
    def stage1_pass(self) -> bool:
        return self.stage1_fraction == 1.0

    @property
    def models_correct(self) -> int:
        return sum(1 for m in self.models.values() if m.match)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage1_fraction"] = self.stage1_fraction
        d["models_correct"] = self.models_correct
        return d


RewardMode = Literal["binary", "model_fraction", "staged"]


@dataclass(frozen=True)
class RewardConfig:
    """How a GradeReport turns into a scalar reward.

    * ``binary``: 1 iff stage 1 passes and every model matches (task-level
      success, the strictest reading of ELT-Bench).
    * ``model_fraction``: official SRDT contribution of this task, i.e. the
      fraction of models that match, gated on stage 1.
    * ``staged`` (default): dense but still execution-derived.
      ``el_weight * stage1_fraction + (1 - el_weight) * stage1_pass * T``
      where ``T`` averages per-model scores: 1 for a full match, otherwise
      ``column_credit * column_fraction`` if the row count is right, else 0.
    """

    mode: RewardMode = "staged"
    el_weight: float = 0.2
    column_credit: float = 0.5
    # Any integrity violation (see ELTReward) zeroes the reward.
    violation_reward: float = 0.0


def compute_reward(report: GradeReport, cfg: RewardConfig, *, skip_el: bool = False) -> float:
    if report.violations:
        return cfg.violation_reward
    n = max(report.n_models, 1)
    if cfg.mode == "binary":
        return float(report.stage1_pass and report.models_correct == report.n_models)
    if cfg.mode == "model_fraction":
        return report.models_correct / n if report.stage1_pass else 0.0
    t = 0.0
    for m in report.models.values():
        if m.match:
            t += 1.0
        elif m.row_count_ok:
            t += cfg.column_credit * m.column_fraction
    t /= n
    if skip_el:
        # Transform-only episodes start from a loaded warehouse; stage 1 is
        # not the policy's doing, so it carries no reward.
        return t
    return cfg.el_weight * report.stage1_fraction + (1 - cfg.el_weight) * float(report.stage1_pass) * t


def report_metrics(report: GradeReport) -> dict[str, float]:
    n = max(report.n_models, 1)
    return {
        "stage1_fraction": report.stage1_fraction,
        "stage1_pass": float(report.stage1_pass),
        "srdt_models_correct": float(report.models_correct),
        "srdt_fraction": report.models_correct / n if report.stage1_pass else 0.0,
        "task_success": float(report.stage1_pass and report.models_correct == report.n_models),
        "column_fraction_mean": (
            sum(m.column_fraction for m in report.models.values()) / n if report.models else 0.0
        ),
        "integrity_violation": float(bool(report.violations)),
    }
