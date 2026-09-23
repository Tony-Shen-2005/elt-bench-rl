"""Extract-and-load stacks.

A stack decides how sources reach the warehouse and what the agent is told
about it; it is orthogonal to the destination.

* ``airbyte``: the official ELT-Bench stack (Airbyte OSS via abctl + the
  ``airbytehq/airbyte`` Terraform provider). Sources run in
  ``elt-docker`` (Postgres, MongoDB, LocalStack S3, a REST API, flat files).
* ``local``: source files live in the workspace and the agent writes its own
  loader. Credential-free; used by the local integration test.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AirbyteStack:
    username: str
    password: str
    workspace_id: str
    api_definition_id: str = ""
    # URL the sandbox uses (ELT-Bench connects the abctl control plane to the
    # elt network under this name) and the URL the host/grader uses.
    server_url: str = "http://airbyte-abctl-control-plane:80/api/public/v1/"
    host_api_url: str = "http://localhost:8000/api/public/v1/"

    @classmethod
    def from_env(cls) -> "AirbyteStack":
        e = os.environ
        return cls(
            username=e["AIRBYTE_USERNAME"],
            password=e["AIRBYTE_PASSWORD"],
            workspace_id=e["AIRBYTE_WORKSPACE_ID"],
            api_definition_id=e.get("AIRBYTE_API_DEFINITION_ID", ""),
            host_api_url=e.get("AIRBYTE_HOST_API_URL", "http://localhost:8000/api/public/v1/"),
        )

    def inject(self, config: dict) -> dict:
        """Fill Airbyte credentials into config.yaml, like ELT-Bench write_config.py."""
        config = json.loads(json.dumps(config))
        ab = config.setdefault("Airbyte", {}).setdefault("config", {})
        ab.update(username=self.username, password=self.password, workspace_id=self.workspace_id,
                  server_url=self.server_url)
        if "custom_api" in config and self.api_definition_id:
            ab["custom_api_definition_id"] = self.api_definition_id
        return config

    # -- grader-side audit ------------------------------------------------------
    def _get(self, path: str) -> dict:
        req = urllib.request.Request(self.host_api_url.rstrip("/") + "/" + path.lstrip("/"))
        token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def succeeded_syncs_into(self, database: str) -> int:
        """Number of succeeded sync jobs whose destination database is ``database``.

        Used to check that raw tables were loaded through Airbyte rather than
        written directly (see docs/DESIGN.md, reward hacking).
        """
        dests = self._get(f"destinations?workspaceIds={self.workspace_id}&limit=1000")["data"]
        dest_ids = {
            d["destinationId"] for d in dests
            if str(d.get("configuration", {}).get("database", "")).upper() == database.upper()
        }
        conns = self._get(f"connections?workspaceIds={self.workspace_id}&limit=1000")["data"]
        n = 0
        for c in conns:
            if c.get("destinationId") in dest_ids:
                jobs = self._get(f"jobs?connectionId={c['connectionId']}&status=succeeded&jobType=sync")
                n += len(jobs.get("data", []))
        return n


STAGE1_AIRBYTE = """# Stage 1: Extraction and Loading (Airbyte + Terraform)
1. Initialize the Airbyte provider in {ws}/elt/main.tf with the username, password and server URL from {ws}/config.yaml (see {ws}/documentation/airbyte_Provider.md), then run `terraform init`. Do not modify the provided code in main.tf.
2. Configure every source and the destination listed in {ws}/config.yaml as Terraform resources in {ws}/elt, following {ws}/documentation. Values in the documentation are examples only; use the real values from config.yaml.
3. Create one connection per source (see {ws}/documentation/connection.md). List exactly the tables from config.yaml in configuration.streams, each with its name and sync_mode.
4. Run `terraform apply`, read the connection IDs from terraform.tfstate, and trigger one sync job per connection with the Airbyte API (see {ws}/documentation/trigger_job.md).{delay}
5. Monitor with `python {ws}/check_job_status.py --server <server> --username <username> --password <password>` until every job has finished. If a job fails, fix the configuration and retry. Do not modify provided files.
"""

STAGE1_LOCAL = """# Stage 1: Extraction and Loading
1. The sources are described under `sources` in {ws}/config.yaml. Their files are under {ws}/sources.
2. Write a loader under {ws}/elt (Python 3 with the `duckdb` package is available) that loads every listed source table into the destination schema, keeping each table's name. Load each row exactly once and preserve the column names.
3. Run it and check the row counts with the `sql` tool.
"""

STAGE2 = """# Stage 2: Transformation
1. Read {ws}/data_model.yaml. For each data model, write one SQL model (a dbt project under {ws}/elt/dbt if dbt is available, otherwise SQL files under {ws}/elt/models) that builds a table with that model's name and exactly the described columns.
2. Source-table schemas are in {ws}/schemas (if present); inspect the loaded tables with the `sql` tool.
3. Build the models in the destination (e.g. `dbt run`), fix errors, and check the results.
4. Call the `submit` tool when every model is built and verified. The warehouse is graded as it is at that moment; nothing you run afterwards counts.
"""


def build_instruction(stack: str, ws: str, destination_notes: str, sync_delay_seconds: int = 0,
                      transform_only: bool = False) -> str:
    parts = [f"We are building an ELT pipeline. The task specification is in {ws}/config.yaml and "
             f"{ws}/data_model.yaml.", destination_notes, ""]
    if transform_only:
        parts.append("Stage 1 has already been done: all raw source tables are loaded in the destination "
                     "schema. Only Stage 2 remains.\n")
    elif stack == "airbyte":
        delay = (f" Wait {sync_delay_seconds} seconds between triggers (warehouse limits); do not wait for a "
                 "job to finish before triggering the next." if sync_delay_seconds else "")
        parts.append(STAGE1_AIRBYTE.format(ws=ws, delay=delay))
    else:
        parts.append(STAGE1_LOCAL.format(ws=ws))
    parts.append(STAGE2.format(ws=ws))
    return "\n".join(parts)


SYSTEM_PROMPT = (
    "You are a data engineer working in a sandboxed Linux workspace. You have three tools: `bash` "
    "runs shell commands, `sql` queries the destination warehouse, and `submit` ends the task. "
    "Work step by step, read files before editing them, check tool outputs for errors, and fix "
    "problems before moving on. Call exactly one tool per message."
)
