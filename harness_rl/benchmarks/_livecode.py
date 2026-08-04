"""LiveCodeBench host-native code execution — NO Docker.

Runs an agent-written `solution.py` against test cases via plain `subprocess`, giving a
genuine checkable reward on a box with no container runtime. Handles both LiveCodeBench
test types:
  - **stdin**: pipe `test["input"]` to `python solution.py`, compare stdout to `test["output"]`.
  - **functional**: LeetCode-style — the solution defines `class Solution` with method
    `fn_name`; a generated driver calls it with the JSON-decoded args and compares the result.

Test-case decoding: LiveCodeBench stores `private_test_cases` as base64(zlib(json)). We decode
defensively (base64→zlib→json, with plain-json fallback).
"""
from __future__ import annotations

import base64
import json
import pickle
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any

from harness_rl.types import Action, Observation, Outcome


def decode_tests(raw: Any) -> list[dict]:
    """Decode a LiveCodeBench test-case field into a list of {input, output, testtype}.

    LiveCodeBench `private_test_cases` are `base64(zlib(pickle(json_str)))`; `public_test_cases`
    are plain JSON. (pickle is the LCB-official format — this is their known dataset.)
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        for decode in (
            lambda s: pickle.loads(zlib.decompress(base64.b64decode(s))),   # private (LCB)
            lambda s: json.loads(zlib.decompress(base64.b64decode(s))),     # zlib+json
            lambda s: json.loads(s),                                        # plain json (public)
        ):
            try:
                out = decode(raw)
                if isinstance(out, str):        # pickle yields the json string
                    out = json.loads(out)
                return out if isinstance(out, list) else []
            except Exception:
                continue
    return []


def _norm(s: str) -> str:
    return "\n".join(line.rstrip() for line in (s or "").strip().splitlines())


_FUNCTIONAL_DRIVER = """
import json, sys
import solution as _sol
_inp = sys.stdin.read().splitlines()
_args = [json.loads(x) for x in _inp if x.strip() != ""]
_obj = _sol.Solution() if hasattr(_sol, "Solution") else _sol
_res = getattr(_obj, {fn!r})(*_args)
print(json.dumps(_res))
"""


def run_tests(workdir: str, tests: list[dict], fn_name: str | None,
              timeout_s: int = 6, max_tests: int = 60) -> tuple[int, int]:
    """Run `solution.py` in workdir against `tests`. Returns (passed, total)."""
    sol = Path(workdir) / "solution.py"
    if not sol.exists() or not tests:
        return 0, len(tests)
    tests = tests[:max_tests]

    driver = Path(workdir) / "_driver.py"
    if fn_name:
        driver.write_text(_FUNCTIONAL_DRIVER.format(fn=fn_name))
        cmd = ["python", str(driver)]
    else:
        cmd = ["python", str(sol)]

    passed = 0
    for t in tests:
        try:
            proc = subprocess.run(cmd, cwd=workdir, input=t.get("input", ""),
                                  capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode != 0:
            continue
        exp, got = t.get("output", ""), proc.stdout
        if _norm(got) == _norm(exp):
            passed += 1
            continue
        # functional outputs: compare JSON-normalized when possible
        if fn_name:
            try:
                if json.loads(got) == json.loads(exp):
                    passed += 1
            except Exception:
                pass
    return passed, len(tests)


class LiveCodeExecEnv:
    """Host-native LiveCodeBench environment. Agent writes `solution.py` (via bash), may run it
    against the PUBLIC tests to iterate, then `TASK_COMPLETE`. `verify()` grades against the
    PRIVATE tests → u_g = passed/total. No container."""

    def __init__(self, public_tests: list[dict], private_tests: list[dict],
                 fn_name: str | None = None, workdir: str | None = None,
                 timeout_s: int = 6, done_tool: str = "submit"):
        self.workdir = workdir or tempfile.mkdtemp(prefix="hrl-lcb-")
        self.public_tests = public_tests
        self.private_tests = private_tests
        self.fn_name = fn_name
        self.timeout_s = timeout_s
        self.done_tool = done_tool
        self._submitted = False

    def reset(self) -> Observation:
        Path(self.workdir).mkdir(parents=True, exist_ok=True)
        ex = self.public_tests[0] if self.public_tests else {}
        hint = (f"Example input:\n{ex.get('input','')}\nExpected output:\n{ex.get('output','')}"
                if ex else "(no public example)")
        mode = f"functional (define class Solution with method {self.fn_name})" if self.fn_name \
            else "stdin/stdout"
        return Observation(text=(
            f"Working dir {self.workdir} (host shell, no container). Write your solution to "
            f"solution.py. Mode: {mode}.\n{hint}"))

    def execute(self, action: Action) -> Observation:
        if action.tool != "bash":
            return Observation(text=f"unsupported tool {action.tool!r}")
        try:
            proc = subprocess.run(["bash", "-lc", action.args.get("cmd", "")], cwd=self.workdir,
                                  capture_output=True, text=True, timeout=self.timeout_s * 4)
            return Observation(text=((proc.stdout or "") + (proc.stderr or ""))[-8000:])
        except subprocess.TimeoutExpired:
            return Observation(text=f"(timeout after {self.timeout_s * 4}s)")

    def is_done(self, action: Action | None) -> bool:
        done = bool(action and action.tool == self.done_tool)
        self._submitted = self._submitted or done
        return done

    def verify(self) -> Outcome:
        passed, total = run_tests(self.workdir, self.private_tests, self.fn_name, self.timeout_s)
        u_g = (passed / total) if total else 0.0
        return Outcome(u_g=u_g, passed_tests=passed, total_tests=total,
                       terminated_reason="submit" if self._submitted else "budget")

    def fork(self) -> "LiveCodeExecEnv":
        """Duplicate live state so a branch continues independently (tree rollout). Copies the
        working dir (solution.py + any agent-created files) into a fresh temp dir; the test lists
        are read-only and shared. `verify()` re-grades each fork's OWN workdir against private tests."""
        import shutil

        new_wd = tempfile.mkdtemp(prefix="hrl-lcb-")
        shutil.copytree(self.workdir, new_wd, dirs_exist_ok=True)
        env = LiveCodeExecEnv(public_tests=self.public_tests, private_tests=self.private_tests,
                              fn_name=self.fn_name, workdir=new_wd, timeout_s=self.timeout_s,
                              done_tool=self.done_tool)
        env._submitted = self._submitted
        return env

    def close(self) -> None:
        pass
