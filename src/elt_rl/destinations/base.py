"""Destination (data warehouse) interface.

Everything warehouse-specific lives behind ``Destination``: namespace
provisioning and isolation, what the agent is told about the warehouse, SQL
execution, identifier case, and eval-SQL rewriting. The environment, tools and
grader only talk to this interface, so adding BigQuery or Redshift means
writing one subclass and registering it in ``destinations/__init__.py``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from elt_rl.task import ELTTask

_RELATION = re.compile(
    r"\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
    flags=re.IGNORECASE,
)


@dataclass
class Namespace:
    """The isolated slice of a warehouse that one rollout owns.

    Concurrent rollouts of the same task (a GRPO group) must not see each
    other's tables, so every rollout gets its own database/file.
    """

    rollout_id: str
    task_name: str
    database: str
    schema: str
    extra: dict = field(default_factory=dict)


class Destination(ABC):
    name: str
    dbt_adapter: str  # pip package the agent uses, e.g. "dbt-snowflake"

    # -- lifecycle -----------------------------------------------------------
    @abstractmethod
    async def provision(self, task: ELTTask, rollout_id: str, *, from_snapshot: bool) -> Namespace:
        """Create an empty (or snapshot-cloned) namespace for one rollout."""

    @abstractmethod
    async def teardown(self, ns: Namespace) -> None: ...

    async def has_snapshot(self, task: ELTTask) -> bool:
        return False

    async def save_snapshot(self, task: ELTTask, ns: Namespace) -> None:
        """Persist ``ns`` (with raw tables loaded) as the task's post-EL snapshot."""
        raise NotImplementedError(f"{self.name} does not support snapshots")

    # -- what the agent sees -------------------------------------------------
    @abstractmethod
    def agent_config(self, ns: Namespace) -> dict:
        """Destination block merged into the agent's config.yaml."""

    @abstractmethod
    def prompt_notes(self, ns: Namespace) -> str:
        """Short, destination-specific instructions appended to the prompt."""

    # -- execution -----------------------------------------------------------
    @abstractmethod
    async def query(self, ns: Namespace, sql: str, *, as_agent: bool) -> pd.DataFrame:
        """Run SQL. ``as_agent=True`` uses the agent's scoped credentials."""

    @abstractmethod
    async def table_row_counts(self, ns: Namespace) -> dict[str, int]:
        """Base tables in the rollout's target schema -> row count (normalized names)."""

    @abstractmethod
    async def table_columns(self, ns: Namespace, table: str) -> list[str]: ...

    @abstractmethod
    def normalize(self, ident: str) -> str:
        """Fold an identifier to the warehouse's canonical case."""

    @abstractmethod
    def qualify(self, ns: Namespace, table: str) -> str:
        """Fully qualified name of ``table`` in the rollout's target schema."""

    def rewrite_eval_sql(self, ns: Namespace, sql: str) -> str:
        """Map benchmark SQL's logical ``<task>.<table>`` to the rollout's namespace."""

        def repl(m: re.Match) -> str:
            kw, schema, table = m.groups()
            if schema.lower() != ns.task_name.lower():
                return m.group(0)
            return f"{kw} {self.qualify(ns, table)}"

        return _RELATION.sub(repl, sql)
