"""Docker-backed environment helper (shared by benchmark adapters).

Executes the fixed `bash` tool inside a per-task container. Docker itself is NOT invoked
at import time; a container is only started in `reset()`. On a no-GPU/no-Docker dev box
these envs simply aren't instantiated (the probe uses EnvStub instead).
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Callable

from harness_rl.types import Action, Observation, Outcome


def docker_available() -> bool:
    return shutil.which("docker") is not None


class DockerEnv:
    """Generic container env: `bash` runs in the container; `verify_fn` scores final state.

    Parameters
    ----------
    image : the task's Docker image (built by the benchmark's own tooling on vast).
    verify_fn : callable(container_name) -> Outcome, running the benchmark's test suite.
    setup_cmds : commands to run after container start (e.g. cd into repo).
    """

    def __init__(
        self,
        image: str,
        verify_fn: Callable[[str], Outcome],
        container_name: str,
        setup_cmds: list[str] | None = None,
        exec_timeout_s: int = 600,
    ) -> None:
        self.image = image
        self.verify_fn = verify_fn
        self.container_name = container_name
        self.setup_cmds = setup_cmds or []
        self.exec_timeout_s = exec_timeout_s
        self._started = False

    def reset(self) -> Observation:
        if not docker_available():
            raise RuntimeError(
                "docker not available — DockerEnv is for the vast.ai runtime. "
                "Use EnvStub for local no-GPU tests."
            )
        subprocess.run(
            ["docker", "run", "-d", "--name", self.container_name, self.image, "sleep", "infinity"],
            check=True, capture_output=True,
        )
        self._started = True
        for cmd in self.setup_cmds:
            self._exec(cmd)
        return Observation(text=f"container {self.container_name} ready ({self.image})")

    def _exec(self, cmd: str) -> str:
        proc = subprocess.run(
            ["docker", "exec", self.container_name, "bash", "-lc", cmd],
            capture_output=True, text=True, timeout=self.exec_timeout_s,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return out[-8000:]  # cap tool output; Gamma decides what to keep

    def execute(self, action: Action) -> Observation:
        if action.tool != "bash":
            return Observation(text=f"unsupported tool {action.tool!r}")
        return Observation(text=self._exec(action.args.get("cmd", "")))

    def is_done(self, action: Action | None) -> bool:
        return bool(action and action.tool == "submit")

    def verify(self) -> Outcome:
        return self.verify_fn(self.container_name)

    def close(self) -> None:
        if self._started:
            subprocess.run(["docker", "rm", "-f", self.container_name],
                           capture_output=True)
            self._started = False
