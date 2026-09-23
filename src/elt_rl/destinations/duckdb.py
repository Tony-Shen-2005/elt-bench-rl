"""DuckDB destination: credential-free, one database file per rollout."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import duckdb
import pandas as pd

from elt_rl.destinations.base import Destination, Namespace
from elt_rl.task import ELTTask


class DuckDBDestination(Destination):
    name = "duckdb"
    dbt_adapter = "dbt-duckdb"

    def __init__(self, root: str | Path, agent_root: str | None = None):
        """
        Args:
            root: host directory holding per-rollout ``.duckdb`` files.
            agent_root: the same directory as seen from inside the sandbox, if
                it is mounted at a different path (Docker). Defaults to ``root``.
        """
        self.root = Path(root)
        self.agent_root = agent_root or str(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, ns: Namespace) -> Path:
        return self.root / f"{ns.database}.duckdb"

    def _snapshot_path(self, task: ELTTask) -> Path:
        # Prefix: DuckDB names the catalog after the file stem, which must not
        # collide with the task schema.
        return self.root / "_snapshots" / f"snap__{task.name}.duckdb"

    async def provision(self, task: ELTTask, rollout_id: str, *, from_snapshot: bool) -> Namespace:
        ns = Namespace(rollout_id, task.name, database=f"{task.name}__{rollout_id}", schema=task.name)
        path = self._path(ns)
        path.unlink(missing_ok=True)
        if from_snapshot:
            shutil.copyfile(self._snapshot_path(task), path)
        else:
            with duckdb.connect(str(path)) as con:
                con.execute(f'CREATE SCHEMA IF NOT EXISTS "{ns.schema}"')
        return ns

    async def teardown(self, ns: Namespace) -> None:
        for suffix in ("", ".wal"):
            Path(str(self._path(ns)) + suffix).unlink(missing_ok=True)

    async def has_snapshot(self, task: ELTTask) -> bool:
        return self._snapshot_path(task).exists()

    async def save_snapshot(self, task: ELTTask, ns: Namespace) -> None:
        dst = self._snapshot_path(task)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + "__tmp.duckdb")
        shutil.copyfile(self._path(ns), tmp)
        # Keep only the raw source tables: a snapshot must never carry a
        # previous rollout's data models into a transform-only episode.
        keep = {self.normalize(t) for t in task.expected_rows}
        with duckdb.connect(str(tmp)) as con:
            rows = con.execute(
                "SELECT table_name, table_type FROM information_schema.tables "
                f"WHERE table_schema = '{ns.schema}'"
            ).fetchall()
            for name, kind in rows:
                if kind != "BASE TABLE" or self.normalize(name) not in keep:
                    obj = "VIEW" if kind == "VIEW" else "TABLE"
                    con.execute(f'DROP {obj} IF EXISTS "{ns.schema}"."{name}" CASCADE')
        tmp.replace(dst)

    def agent_config(self, ns: Namespace) -> dict:
        return {
            "duckdb": {
                "config": {
                    "path": f"{self.agent_root}/{ns.database}.duckdb",
                    "schema": ns.schema,
                }
            }
        }

    def prompt_notes(self, ns: Namespace) -> str:
        path = self.agent_config(ns)["duckdb"]["config"]["path"]
        return (
            f"The destination warehouse is DuckDB, database file `{path}`. "
            f"Load raw tables AND create the final data models in schema `{ns.schema}`. "
            "Only one process can write to the file at a time, so close connections "
            "when done. Use the `sql` tool to inspect tables."
        )

    def _run(self, ns: Namespace, sql: str) -> pd.DataFrame:
        with duckdb.connect(str(self._path(ns))) as con:
            cur = con.execute(sql)
            if cur.description is None:
                return pd.DataFrame()
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)

    async def query(self, ns: Namespace, sql: str, *, as_agent: bool) -> pd.DataFrame:
        return await asyncio.to_thread(self._run, ns, sql)

    async def table_row_counts(self, ns: Namespace) -> dict[str, int]:
        df = await self.query(
            ns,
            "SELECT table_name, estimated_size FROM duckdb_tables() "
            f"WHERE schema_name = '{ns.schema}'",
            as_agent=False,
        )
        counts = {}
        for name in df["table_name"] if not df.empty else []:
            # estimated_size is exact for DuckDB base tables but cheap to verify.
            n = await self.query(ns, f"SELECT COUNT(*) FROM {self.qualify(ns, name)}", as_agent=False)
            counts[self.normalize(name)] = int(n.iloc[0, 0])
        return counts

    async def table_columns(self, ns: Namespace, table: str) -> list[str]:
        df = await self.query(
            ns,
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_schema = '{ns.schema}' AND lower(table_name) = lower('{table}')",
            as_agent=False,
        )
        return list(df["column_name"]) if not df.empty else []

    def normalize(self, ident: str) -> str:
        return ident.lower()

    def qualify(self, ns: Namespace, table: str) -> str:
        return f'"{ns.schema}"."{table}"'
