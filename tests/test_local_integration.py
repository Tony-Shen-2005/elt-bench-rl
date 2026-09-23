"""Credential-free end-to-end test: real env loop, tools, sandbox, DuckDB and grader.

Only the model is scripted (see tests/scripted.py). Covers: a correct
pipeline scores 1.0, partial pipelines get partial credit, concurrent group
rollouts are isolated, snapshots enable transform-only episodes, and several
reward-hacking attempts score nothing.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import duckdb
import pytest

from tinker_cookbook.rl.rollouts import do_group_rollout, do_single_rollout

from elt_rl.destinations.duckdb import DuckDBDestination
from elt_rl.env import ELTEnvGroupBuilder
from elt_rl.task import load_local_task
from scripted import GroupPolicy, ScriptBook, ScriptRenderer, call

FIX = Path(__file__).parent / "fixtures"
SOL = FIX / "tiny_shop_solution"
LOAD = (SOL / "load.py").read_text()
TRANSFORM = (SOL / "transform.py").read_text()


def write(path: str, content: str) -> dict:
    return call("bash", command=f"mkdir -p $(dirname {path}) && cat > {path} <<'PYEOF'\n{content}\nPYEOF")


ORACLE = [
    call("bash", command="cat config.yaml data_model.yaml"),
    write("elt/load.py", LOAD),
    call("bash", command="python elt/load.py"),
    call("sql", query="select count(*) as n from orders"),
    write("elt/models/transform.py", TRANSFORM),
    call("bash", command="python elt/models/transform.py"),
    call("sql", query="select * from customer_ltv order by customer_id"),
    call("submit"),
]
TRANSFORM_ONLY = [write("elt/models/transform.py", TRANSFORM),
                  call("bash", command="python elt/models/transform.py"), call("submit")]


@pytest.fixture
def task():
    return load_local_task(FIX / "tiny_shop")


def make_builder(task, tmp_path, book, group_size=1, **kw):
    return ELTEnvGroupBuilder(task=task, destination=DuckDBDestination(tmp_path / "wh"),
                              model_name="scripted", group_size=group_size,
                              renderer=ScriptRenderer(book), max_turns=20, command_timeout=60, **kw)


async def run_scripts(builder, book, scripts):
    policies = [book.add(s) for s in scripts]
    envs = await builder.make_envs()
    try:
        trajs = await asyncio.gather(*(do_single_rollout(p, e) for p, e in zip(policies, envs)))
        return trajs, list(builder.rewards)
    finally:
        await builder.cleanup()


def final_reward(traj) -> float:
    return traj.transitions[-1].reward


def test_workspace_contains_spec_but_no_grader_files(task, tmp_path):
    book = ScriptBook()
    ls = call("bash", command="find . -type f | sort; grep -rl customer_ltv . || true")

    async def go():
        b = make_builder(task, tmp_path, book)
        envs = await b.make_envs()
        env = envs[0]
        obs, _ = await env.initial_observation()
        ws = Path(b._live[0][0].workspace)
        files = sorted(str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file())
        await b.cleanup()
        return files

    files = asyncio.run(go())
    assert "config.yaml" in files and "data_model.yaml" in files
    assert any(f.startswith("sources/") for f in files)
    assert not any("grader" in f or f.endswith(".csv") and "gt" in f for f in files)
    assert "expected_rows.json" not in " ".join(files)


def test_oracle_partial_and_hacks_in_one_group(task, tmp_path):
    book = ScriptBook()
    scripts = {
        "oracle": ORACLE,
        "el_only": ORACLE[:4] + [call("submit")],
        "submit_immediately": [call("submit")],
        # Start the correct pipeline in the background, then submit at once,
        # hoping it finishes before/while grading. Background work is killed
        # before grading, so the warehouse is empty.
        "background_writer": [
            write("elt/load.py", LOAD), write("elt/models/transform.py", TRANSFORM),
            call("bash", command="nohup bash -c 'sleep 3; python elt/load.py; python elt/models/transform.py' >/dev/null 2>&1 &"),
            call("submit"),
        ],
        # Right row counts, garbage content: at most the stage-1 share.
        "fake_raw_tables": [
            call("bash", command=(
                "python - <<'PYEOF'\nimport duckdb, yaml\nc = yaml.safe_load(open('config.yaml'))['duckdb']['config']\n"
                "con = duckdb.connect(c['path'])\n"
                "for t, n in [('customers', 8), ('orders', 20), ('products', 5)]:\n"
                "    con.execute(f'CREATE TABLE tiny_shop.{t} AS SELECT range AS x FROM range({n})')\n"
                "con.execute('CREATE TABLE tiny_shop.customer_ltv AS SELECT range AS customer_id FROM range(8)')\n"
                "con.close()\nPYEOF")),
            call("submit"),
        ],
    }
    b = make_builder(task, tmp_path, book, group_size=len(scripts), log_dir=str(tmp_path / "logs"))
    trajs, rewards = asyncio.run(run_scripts(b, book, list(scripts.values())))
    got = dict(zip(scripts, (final_reward(t) for t in trajs)))

    assert got["oracle"] == pytest.approx(1.0)
    assert got["el_only"] == pytest.approx(0.2)  # stage-1 share only
    assert got["submit_immediately"] == 0.0
    assert got["background_writer"] == 0.0
    assert got["fake_raw_tables"] <= 0.2 + 1e-9
    oracle_metrics = trajs[0].transitions[-1].metrics
    assert oracle_metrics["task_success"] == 1.0 and oracle_metrics["srdt_models_correct"] == 2.0
    # Namespaces are isolated and torn down.
    assert not list((tmp_path / "wh").glob("tiny_shop__*.duckdb"))
    assert len(list((tmp_path / "logs").glob("*.json"))) == len(scripts)


def test_background_writer_really_is_dead(task, tmp_path):
    """After grading, the killed background job must not resurrect the tables."""
    book = ScriptBook()
    script = [write("elt/load.py", LOAD),
              call("bash", command="nohup bash -c 'sleep 2; python elt/load.py' >/dev/null 2>&1 &"),
              call("submit")]

    async def go():
        b = make_builder(task, tmp_path, book)
        p = book.add(script)
        env = (await b.make_envs())[0]
        traj = await do_single_rollout(p, env)
        sandbox, ns = b._live[0]
        await asyncio.sleep(3)
        path = b.destination._path(ns)
        with duckdb.connect(str(path)) as con:
            n = con.execute("select count(*) from information_schema.tables where table_schema='tiny_shop'").fetchone()[0]
        await b.cleanup()
        return final_reward(traj), n

    reward, n_tables = asyncio.run(go())
    assert reward == 0.0 and n_tables == 0


def test_group_rollout_via_tinker_and_transform_only_snapshot(task, tmp_path):
    """Uses tinker's do_group_rollout (the path RL training takes). The first
    successful rollout harvests a post-EL snapshot; a transform-only episode
    then starts from it, with raw tables present and no data models."""
    book = ScriptBook()
    sid = len(book.scripts)
    book.add(ORACLE)
    b = make_builder(task, tmp_path, book, group_size=3)
    group = asyncio.run(do_group_rollout(b, GroupPolicy(sid, n_msgs_initial=2)))
    assert [final_reward(t) for t in group.trajectories_G] == pytest.approx([1.0, 1.0, 1.0])

    dest = b.destination
    snap = dest._snapshot_path(task)
    assert snap.exists()
    with duckdb.connect(str(snap)) as con:
        tables = {r[0] for r in con.execute("select table_name from information_schema.tables").fetchall()}
    assert tables == {"customers", "orders", "products"}  # models pruned

    b2 = make_builder(task, tmp_path, book, transform_only=True)
    trajs, _ = asyncio.run(run_scripts(b2, book, [TRANSFORM_ONLY, [call("submit")]][:1]))
    assert final_reward(trajs[0]) == pytest.approx(1.0)
    b3 = make_builder(task, tmp_path, book, transform_only=True)
    trajs, _ = asyncio.run(run_scripts(b3, book, [[call("submit")]]))
    assert final_reward(trajs[0]) == 0.0  # raw tables alone earn nothing in transform-only mode
