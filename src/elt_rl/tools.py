"""Agent tools: bash in the sandbox, SQL against the rollout's namespace, submit."""

from __future__ import annotations

import json
from typing import Annotated

from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

from elt_rl.destinations import Destination, Namespace

MAX_OUTPUT_CHARS = 8192
MAX_SQL_ROWS = 50


def _clip(s: str, n: int = MAX_OUTPUT_CHARS) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n...[{len(s) - n} chars truncated]...\n" + s[-n // 2 :]


class ELTTools:
    """Stateful per-rollout tool set."""

    def __init__(self, sandbox, destination: Destination, ns: Namespace, command_timeout: int = 600):
        self.sandbox = sandbox
        self.destination = destination
        self.ns = ns
        self.command_timeout = command_timeout
        self.submitted = False
        self.n_bash = 0
        self.n_sql = 0

    @tool
    async def bash(self, command: Annotated[str, "The bash command to run in the workspace."]) -> ToolResult:
        """Run a bash command in the sandbox (terraform, dbt, python, psql, curl, file edits...)."""
        self.n_bash += 1
        r = await self.sandbox.run_command(command, timeout=self.command_timeout,
                                           max_output_bytes=MAX_OUTPUT_CHARS)
        out = json.dumps({"exit_code": r.exit_code, "stdout": _clip(r.stdout), "stderr": _clip(r.stderr)})
        return simple_tool_result(out, metrics={"bash_error": float(r.exit_code != 0)})

    @tool
    async def sql(self, query: Annotated[str, "A single SQL statement for the destination warehouse."]) -> ToolResult:
        """Run SQL against the destination warehouse with your pipeline's credentials.

        Returns at most 50 rows. Use it to inspect loaded raw tables and verify
        your data models before submitting.
        """
        self.n_sql += 1
        try:
            df = await self.destination.query(self.ns, query, as_agent=True)
        except Exception as e:  # warehouse errors are feedback, not crashes
            return simple_tool_result(f"ERROR: {_clip(str(e), 2000)}", metrics={"sql_error": 1.0})
        text = df.head(MAX_SQL_ROWS).to_string(index=False, max_colwidth=60)
        if len(df) > MAX_SQL_ROWS:
            text += f"\n... ({len(df)} rows total, showing {MAX_SQL_ROWS})"
        return simple_tool_result(_clip(text), metrics={"sql_error": 0.0})

    @tool
    async def submit(self) -> ToolResult:
        """Submit the pipeline. The warehouse is graded as it is now; the episode ends."""
        self.submitted = True
        return simple_tool_result("Submitted. The warehouse state will now be graded.", should_stop=True)

    def all(self) -> list:
        return [self.bash, self.sql, self.submit]
