import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elt_rl.grading import GradeReport, ModelResult, RewardConfig, compare_table, compute_reward


def gt():
    return pd.DataFrame({"id": ["1", "2", "3"], "name": ["a", "b", "c"], "amt": ["1.00", "2.5", None]})


def test_exact_match_any_order_and_case():
    cand = pd.DataFrame({"AMT": [2.5, 1.0, np.nan], "NAME": ["B", "a", "c"], "ID": [2, 1, 3], "extra": [0, 0, 0]})
    r = compare_table(gt(), cand, ["id"])
    assert r.match and r.row_count_ok and r.column_fraction == 1.0


def test_numeric_tolerance_is_one_percent():
    ok = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "amt": [1.009, 2.5, None]})
    bad = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "amt": [1.02, 2.5, None]})
    assert compare_table(gt(), ok, ["id"]).match
    assert compare_table(gt(), bad, ["id"]).unmatched == ["amt"]


def test_null_vs_value_fails():
    cand = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "amt": [1.0, 2.5, 0.0]})
    assert not compare_table(gt(), cand, ["id"]).match


def test_missing_column_and_row_count():
    r = compare_table(gt(), pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}), ["id"])
    assert r.missed == ["amt"] and r.column_fraction == pytest.approx(2 / 3)
    r = compare_table(gt(), pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "amt": [1, 2.5]}), ["id"])
    assert not r.row_count_ok and not r.match


def _report(stage1_ok: bool, models: dict[str, ModelResult]) -> GradeReport:
    return GradeReport(tables={"t": (5 if stage1_ok else 4, 5)}, models=models, n_models=2)


def test_reward_modes():
    full = ModelResult(exists=True, row_count_ok=True, match=True, matched=["a", "b"])
    half = ModelResult(exists=True, row_count_ok=True, matched=["a"], unmatched=["b"])
    rep = _report(True, {"m1": full, "m2": half})
    assert compute_reward(rep, RewardConfig(mode="binary")) == 0.0
    assert compute_reward(rep, RewardConfig(mode="model_fraction")) == 0.5
    # staged: 0.2 * 1 + 0.8 * (1 + 0.5 * 0.5) / 2
    assert compute_reward(rep, RewardConfig()) == pytest.approx(0.2 + 0.8 * 0.625)
    assert compute_reward(rep, RewardConfig(), skip_el=True) == pytest.approx(0.625)
    # stage 1 gates stage 2
    assert compute_reward(_report(False, {"m1": full, "m2": full}), RewardConfig()) == 0.0
    rep.violations = ["x"]
    assert compute_reward(rep, RewardConfig()) == 0.0


@pytest.mark.skipif(not os.environ.get("ELT_BENCH_REPO"), reason="set ELT_BENCH_REPO for parity check")
def test_parity_with_official_comparator():
    """Our comparator agrees with ELT-Bench's check_corretness on the same inputs."""
    sys.path.insert(0, str(Path(os.environ["ELT_BENCH_REPO"]) / "evaluation"))
    import eva_stage2 as official

    cases = [
        pd.DataFrame({"AMT": [2.5, 1.0, np.nan], "NAME": ["B", "a", "c"], "ID": [2, 1, 3]}),
        pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "amt": [1.02, 2.5, None]}),
        pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "amt": [1.0, 2.5, 0.0]}),
        pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
    ]
    for cand in cases:
        # Official path: CSV round trip, dtype=str, sort by keys.
        g = official.sort_by_keys(gt(), ["id"])
        c = official.sort_by_keys(pd.read_csv(pd.io.common.StringIO(cand.to_csv(index=False)), dtype=str), ["id"])
        assert official.check_corretness(g, c)["match"] == compare_table(gt(), cand, ["id"]).match
