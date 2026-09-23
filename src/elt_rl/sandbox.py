"""Sandboxes implementing tinker_cookbook's ``SandboxInterface``.

* ``DockerSandbox``: one container per rollout from ELT-Bench's ``elt-swe``
  image (Terraform, dbt adapters, psql, pymongo, awscli), attached to the
  ``elt-docker_elt_network`` so it can reach the sources and Airbyte. Only the
  rollout's workspace is mounted, so grader files are unreachable.
* ``LocalSandbox``: a subprocess in a temp workspace. No isolation: for
  credential-free tests and debugging only, never for training.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import signal
import tempfile
import uuid
from pathlib import Path

from tinker_cookbook.sandbox import SandboxResult

DEFAULT_MAX_OUTPUT = 128 * 1024


def _truncate(b: bytes, limit: int | None) -> str:
    limit = limit or DEFAULT_MAX_OUTPUT
    s = b.decode("utf-8", errors="replace")
    if len(s) > limit:
        half = limit // 2
        s = s[:half] + f"\n... [{len(s) - limit} chars truncated] ...\n" + s[-half:]
    return s


async def _exec(argv: list[str], timeout: int, max_output: int | None,
                pgids: list[int] | None = None, **kw) -> SandboxResult:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True, **kw,
    )
    if pgids is not None:
        pgids.append(proc.pid)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = await proc.communicate()
        return SandboxResult(
            stdout=_truncate(out, max_output),
            stderr=_truncate(err, max_output) + f"\n[command timed out after {timeout}s]",
            exit_code=124,
        )
    return SandboxResult(
        stdout=_truncate(out, max_output), stderr=_truncate(err, max_output),
        exit_code=proc.returncode or 0,
    )


class LocalSandbox:
    """Runs commands with ``bash -c`` in a private workspace directory."""

    def __init__(self, workspace: Path | None = None, env: dict[str, str] | None = None):
        self.workspace = Path(workspace or tempfile.mkdtemp(prefix="eltrl-ws-"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._env = {**os.environ, **(env or {}), "WORKSPACE": str(self.workspace)}
        self._id = uuid.uuid4().hex[:8]
        self._pgids: list[int] = []

    @property
    def sandbox_id(self) -> str:
        return f"local-{self._id}"

    @property
    def workspace_path(self) -> str:
        return str(self.workspace)

    def _resolve(self, path: str) -> Path:
        # Paths under the canonical /workspace prefix map to our directory.
        if path == "/workspace" or path.startswith("/workspace/"):
            return self.workspace / path.removeprefix("/workspace").lstrip("/")
        p = Path(path)
        return p if p.is_absolute() else self.workspace / p

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60,
                          max_output_bytes: int | None = None) -> SandboxResult:
        cwd = self._resolve(workdir) if workdir else self.workspace
        return await _exec(["bash", "-c", command], timeout, max_output_bytes, self._pgids,
                           cwd=cwd, env=self._env)

    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
        p = self._resolve(path)
        try:
            data = p.read_bytes()[: max_bytes or None]
        except OSError as e:
            return SandboxResult(stdout="", stderr=str(e), exit_code=1)
        return SandboxResult(stdout=data.decode("utf-8", errors="replace"), stderr="", exit_code=0)

    async def write_file(self, path: str, content: str | bytes, executable: bool = False,
                         timeout: int = 60) -> SandboxResult:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode() if isinstance(content, str) else content)
        if executable:
            p.chmod(0o755)
        return SandboxResult(stdout="", stderr="", exit_code=0)

    async def kill_background(self) -> None:
        # Each command runs in its own process group; anything it left
        # running in the background (`cmd &`) is still in that group.
        for pgid in self._pgids:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self._pgids.clear()

    async def cleanup(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)


class DockerSandbox:
    """One long-lived container per rollout; commands run via ``docker exec``."""

    def __init__(self, container: str, host_workspace: Path):
        self.container = container
        self.host_workspace = host_workspace

    workspace_path = "/workspace"

    @classmethod
    async def create(
        cls,
        image: str = "elt-swe",
        network: str | None = "elt-docker_elt_network",
        extra_mounts: dict[str, str] | None = None,
        name_prefix: str = "eltrl",
        tf_plugin_cache: Path | None = None,
    ) -> "DockerSandbox":
        host_ws = Path(tempfile.mkdtemp(prefix="eltrl-ws-"))
        name = f"{name_prefix}-{uuid.uuid4().hex[:10]}"
        argv = ["docker", "run", "-d", "--rm", "--name", name, "-v", f"{host_ws}:/workspace",
                "-w", "/workspace"]
        if tf_plugin_cache is not None:
            # Shared provider cache so each rollout's `terraform init` skips the
            # ~100MB airbyte provider download. Terraform does not guarantee the
            # cache is safe under concurrent writes: warm it with one rollout
            # before launching a parallel batch, after which inits only read it.
            tf_plugin_cache.mkdir(parents=True, exist_ok=True)
            argv += ["-v", f"{tf_plugin_cache}:/tf-plugin-cache",
                     "-e", "TF_PLUGIN_CACHE_DIR=/tf-plugin-cache"]
        if network:
            argv += ["--network", network]
        for host, ctr in (extra_mounts or {}).items():
            argv += ["-v", f"{host}:{ctr}"]
        argv += [image, "sleep", "infinity"]
        res = await _exec(argv, 120, None)
        if res.exit_code != 0:
            raise RuntimeError(f"docker run failed: {res.stderr}")
        return cls(name, host_ws)

    @property
    def sandbox_id(self) -> str:
        return self.container

    async def send_heartbeat(self, timeout: int = 30) -> None:
        return None

    async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60,
                          max_output_bytes: int | None = None) -> SandboxResult:
        # `timeout` inside the container so a hung command dies there too.
        argv = ["docker", "exec", "-w", workdir or "/workspace", self.container,
                "timeout", "-s", "KILL", str(timeout), "bash", "-c", command]
        res = await _exec(argv, timeout + 15, max_output_bytes)
        if res.exit_code == 137:
            res = SandboxResult(stdout=res.stdout, stderr=res.stderr + f"\n[command timed out after {timeout}s]",
                                exit_code=124)
        return res

    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult:
        cmd = f"head -c {max_bytes} {shlex.quote(path)}" if max_bytes else f"cat {shlex.quote(path)}"
        return await self.run_command(cmd, timeout=timeout)

    async def write_file(self, path: str, content: str | bytes, executable: bool = False,
                         timeout: int = 60) -> SandboxResult:
        data = content.encode() if isinstance(content, str) else content
        cmd = f"mkdir -p $(dirname {shlex.quote(path)}) && cat > {shlex.quote(path)}"
        if executable:
            cmd += f" && chmod +x {shlex.quote(path)}"
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", "-i", self.container, "bash", "-c", cmd,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(data), timeout=timeout)
        return SandboxResult(stdout=out.decode(), stderr=err.decode(), exit_code=proc.returncode or 0)

    async def kill_background(self) -> None:
        # Kill every process except PID 1 (`sleep infinity`), so nothing the
        # agent left running can change the warehouse after submission.
        await _exec(["docker", "exec", self.container, "bash", "-c", "kill -9 -1 || true"], 30, None)

    async def cleanup(self) -> None:
        await _exec(["docker", "rm", "-f", self.container], 60, None)
        shutil.rmtree(self.host_workspace, ignore_errors=True)
