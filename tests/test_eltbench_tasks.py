"""Official-task plumbing that needs no credentials: loading, the agent's
config.yaml, eval-SQL rewriting, and grader/agent file separation."""

import os
from pathlib import Path

import pytest

from elt_rl.destinations import Namespace
from elt_rl.destinations.snowflake import SnowflakeDestination
from elt_rl.env import workspace_config
from elt_rl.stacks import AirbyteStack, build_instruction
from elt_rl.task import load_eltbench_tasks

REPO = os.environ.get("ELT_BENCH_REPO")
pytestmark = pytest.mark.skipif(not REPO, reason="set ELT_BENCH_REPO to an ELT-Bench checkout")


@pytest.fixture(scope="module")
def tasks():
    return {t.name: t for t in load_eltbench_tasks(REPO, Path(REPO) / "ground_truth" / "gt_snowflake")}


def test_all_100_tasks_load(tasks):
    assert len(tasks) == 100
    assert sum(len(t.models) for t in tasks.values()) >= 200
    amp = tasks["amplitude"]
    assert amp.expected_rows == {"event": 11, "event_type": 5} or sum(amp.expected_rows.values()) == 16
    assert "schemas/event.csv" in amp.workspace_files and "elt/main.tf" in amp.workspace_files


def test_grader_files_never_reach_workspace(tasks):
    for t in tasks.values():
        for rel, src in t.workspace_files.items():
            s = str(src)
            assert "/evaluation/" not in s and "ground_truth" not in s and "gt_" not in rel, (t.name, rel)


def test_rollout_config_and_eval_sql(tasks):
    t = tasks["amplitude"]
    dest = SnowflakeDestination(account="acct", admin_user="admin", admin_password="x")
    ns = Namespace("ab12cd34", t.name, database=dest._db(t.name, "ab12cd34"), schema="AIRBYTE_SCHEMA")
    ab = AirbyteStack(username="u", password="p", workspace_id="w", api_definition_id="api")
    cfg = workspace_config(t, dest, ns, ab)
    assert cfg["snowflake"]["config"]["database"] == "ELTRL_AMPLITUDE_AB12CD34"
    assert cfg["snowflake"]["config"]["username"] == "AIRBYTE_USER"  # agent creds, not admin
    assert "admin" not in str(cfg)
    assert cfg["Airbyte"]["config"]["custom_api_definition_id"] == "api"
    assert set(cfg) >= {"postgres", "custom_api", "Airbyte", "snowflake"}

    sql = "select * from amplitude.amplitude__sessions s join amplitude.event e on 1=1 order by 1"
    out = dest.rewrite_eval_sql(ns, sql)
    assert "from ELTRL_AMPLITUDE_AB12CD34.AIRBYTE_SCHEMA.amplitude__sessions" in out
    assert "join ELTRL_AMPLITUDE_AB12CD34.AIRBYTE_SCHEMA.event" in out

    prompt = build_instruction("airbyte", "/workspace", dest.prompt_notes(ns))
    assert "terraform apply" in prompt and "submit" in prompt and "sleep" not in prompt
