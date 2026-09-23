"""Snowflake destination (the original ELT-Bench warehouse).

Layout follows ELT-Bench: tables live in ``<database>.AIRBYTE_SCHEMA``. Each
rollout gets its own database ``ELTRL_<TASK>_<ROLLOUT>`` so a GRPO group can
run concurrently; the benchmark's eval SQL is rewritten to point there.

Two credentials:
* ``admin``: used by the environment to create/drop databases and to grade
  (e.g. SYSADMIN). Never shown to the agent.
* ``agent``: the ELT-Bench ``AIRBYTE_USER`` / ``AIRBYTE_ROLE`` from
  ``setup/destination/setup.sql``; written into the agent's config.yaml and
  used by Airbyte and dbt.
"""

from __future__ import annotations

import asyncio
import re

import pandas as pd

from elt_rl.destinations.base import Destination, Namespace
from elt_rl.task import ELTTask

SCHEMA = "AIRBYTE_SCHEMA"


def _ident(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s).upper()
    return s


class SnowflakeDestination(Destination):
    name = "snowflake"
    dbt_adapter = "dbt-snowflake"

    def __init__(
        self,
        account: str,
        admin_user: str,
        admin_password: str,
        admin_role: str = "SYSADMIN",
        warehouse: str = "AIRBYTE_WAREHOUSE",
        agent_user: str = "AIRBYTE_USER",
        agent_password: str = "Snowflake@123",
        agent_role: str = "AIRBYTE_ROLE",
        database_prefix: str = "ELTRL",
    ):
        self.account = account
        self.admin = dict(user=admin_user, password=admin_password, role=admin_role)
        self.agent = dict(user=agent_user, password=agent_password, role=agent_role)
        self.warehouse = warehouse
        self.prefix = database_prefix

    @classmethod
    def from_env(cls) -> "SnowflakeDestination":
        import os

        e = os.environ
        return cls(
            account=e["SNOWFLAKE_ACCOUNT"],
            admin_user=e["SNOWFLAKE_ADMIN_USER"],
            admin_password=e["SNOWFLAKE_ADMIN_PASSWORD"],
            admin_role=e.get("SNOWFLAKE_ADMIN_ROLE", "SYSADMIN"),
            warehouse=e.get("SNOWFLAKE_WAREHOUSE", "AIRBYTE_WAREHOUSE"),
            agent_user=e.get("SNOWFLAKE_AGENT_USER", "AIRBYTE_USER"),
            agent_password=e.get("SNOWFLAKE_AGENT_PASSWORD", "Snowflake@123"),
            agent_role=e.get("SNOWFLAKE_AGENT_ROLE", "AIRBYTE_ROLE"),
        )

    # -- connections ----------------------------------------------------------
    def _connect(self, as_agent: bool):
        import snowflake.connector

        creds = self.agent if as_agent else self.admin
        return snowflake.connector.connect(account=self.account, warehouse=self.warehouse, **creds)

    def _run(self, sql: str | list[str], as_agent: bool) -> pd.DataFrame:
        stmts = [sql] if isinstance(sql, str) else sql
        with self._connect(as_agent) as con, con.cursor() as cur:
            df = pd.DataFrame()
            for s in stmts:
                cur.execute(s)
                if cur.description:
                    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
            return df

    async def _arun(self, sql: str | list[str], as_agent: bool = False) -> pd.DataFrame:
        return await asyncio.to_thread(self._run, sql, as_agent)

    # -- lifecycle ------------------------------------------------------------
    def _db(self, task_name: str, rollout_id: str) -> str:
        return _ident(f"{self.prefix}_{task_name}_{rollout_id}")

    def _snapshot_db(self, task: ELTTask) -> str:
        return _ident(f"{self.prefix}_SNAP_{task.name}")

    def _grant_sql(self, db: str) -> list[str]:
        role = self.agent["role"]
        return [
            f"GRANT USAGE, CREATE SCHEMA ON DATABASE {db} TO ROLE {role}",
            f"GRANT OWNERSHIP ON SCHEMA {db}.{SCHEMA} TO ROLE {role} COPY CURRENT GRANTS",
            f"GRANT ALL ON ALL TABLES IN SCHEMA {db}.{SCHEMA} TO ROLE {role}",
        ]

    async def provision(self, task: ELTTask, rollout_id: str, *, from_snapshot: bool) -> Namespace:
        db = self._db(task.name, rollout_id)
        if from_snapshot:
            # Zero-copy clone: O(seconds) regardless of data size.
            create = [f"CREATE OR REPLACE DATABASE {db} CLONE {self._snapshot_db(task)}"]
        else:
            create = [f"CREATE OR REPLACE DATABASE {db}", f"CREATE SCHEMA {db}.{SCHEMA}"]
        await self._arun(create + self._grant_sql(db))
        return Namespace(rollout_id, task.name, database=db, schema=SCHEMA)

    async def teardown(self, ns: Namespace) -> None:
        await self._arun(f"DROP DATABASE IF EXISTS {ns.database}")

    async def has_snapshot(self, task: ELTTask) -> bool:
        df = await self._arun(f"SHOW DATABASES LIKE '{self._snapshot_db(task)}'")
        return not df.empty

    async def save_snapshot(self, task: ELTTask, ns: Namespace) -> None:
        snap = self._snapshot_db(task)
        await self._arun(f"CREATE OR REPLACE DATABASE {snap} CLONE {ns.database}")
        # Keep only raw source tables (no data models from this rollout).
        keep = {self.normalize(t) for t in task.expected_rows}
        df = await self._arun(
            f"SELECT table_name, table_type FROM {snap}.information_schema.tables "
            f"WHERE table_schema = '{SCHEMA}'"
        )
        drops = [
            f"DROP {'VIEW' if r.TABLE_TYPE == 'VIEW' else 'TABLE'} IF EXISTS {snap}.{SCHEMA}.\"{r.TABLE_NAME}\""
            for r in df.itertuples()
            if r.TABLE_TYPE != "BASE TABLE" or self.normalize(r.TABLE_NAME) not in keep
        ]
        if drops:
            await self._arun(drops)

    # -- agent view -----------------------------------------------------------
    def agent_config(self, ns: Namespace) -> dict:
        return {
            "snowflake": {
                "config": {
                    "account": self.account,
                    "database": ns.database,
                    "schema": ns.schema,
                    "role": self.agent["role"],
                    "username": self.agent["user"],
                    "password": self.agent["password"],
                    "warehouse": self.warehouse,
                }
            }
        }

    def prompt_notes(self, ns: Namespace) -> str:
        return (
            f"The destination warehouse is Snowflake. Raw tables must be loaded into "
            f"`{ns.database}.{ns.schema}` and the final data models must be created in the "
            f"same schema. Connection details are under `snowflake` in /workspace/config.yaml; "
            f"use dbt-snowflake for transformations and the `sql` tool to inspect tables."
        )

    # -- execution ------------------------------------------------------------
    async def query(self, ns: Namespace, sql: str, *, as_agent: bool) -> pd.DataFrame:
        use = f"USE SCHEMA {ns.database}.{ns.schema}"
        return await self._arun([use, sql], as_agent=as_agent)

    async def table_row_counts(self, ns: Namespace) -> dict[str, int]:
        # One metadata query instead of N COUNT(*)s; ROW_COUNT is exact for
        # base tables in Snowflake.
        df = await self._arun(
            f"SELECT table_name, row_count FROM {ns.database}.information_schema.tables "
            f"WHERE table_schema = '{ns.schema}' AND table_type = 'BASE TABLE'"
        )
        return {self.normalize(r.TABLE_NAME): int(r.ROW_COUNT or 0) for r in df.itertuples()}

    async def table_columns(self, ns: Namespace, table: str) -> list[str]:
        df = await self._arun(
            f"SELECT column_name FROM {ns.database}.information_schema.columns "
            f"WHERE table_schema = '{ns.schema}' AND table_name = '{self.normalize(table)}'"
        )
        return list(df.iloc[:, 0]) if not df.empty else []

    def normalize(self, ident: str) -> str:
        return ident.upper()

    def qualify(self, ns: Namespace, table: str) -> str:
        return f"{ns.database}.{ns.schema}.{table}"
