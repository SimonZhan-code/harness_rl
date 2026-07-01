"""The agent control loop — a mini-swe-agent-style harness with a Gamma seam.

Difference from vanilla mini-swe-agent: instead of a naive linear message append, the
context handed to the policy is produced by a pluggable `ContextManager` (Gamma). The
tool set and parser (Omega_0) are FIXED. The loop is reused verbatim in Step 2 RL — only
the `ModelClient.base_url` changes.

Parser (fixed): the model emits exactly one fenced ```bash ... ``` block to run a shell
command, or a line `TASK_COMPLETE` to submit. Malformed output → a reminder observation.
"""
from __future__ import annotations

import re

from harness_rl.benchmarks.base import Environment
from harness_rl.gamma.base import BaseContextManager, count_tokens
from harness_rl.serving.client import ModelClient
from harness_rl.types import (
    Action,
    BudgetCaps,
    GammaSnapshot,
    Observation,
    Outcome,
    Step,
    TaskSpec,
    Trace,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent operating in a sandboxed environment. You may run shell "
    "commands to inspect and modify the project. Respond with EXACTLY ONE action per "
    "turn:\n"
    "  - to run a command, output a single fenced block:\n"
    "    ```bash\n    <your command>\n    ```\n"
    "  - when the task is fully solved, output the single line: TASK_COMPLETE\n"
    "Think briefly, then act. Do not output multiple bash blocks in one turn."
)

_BASH_RE = re.compile(r"```bash\s*\n(.*?)```", re.DOTALL)
_DONE_RE = re.compile(r"^\s*TASK_COMPLETE\s*$", re.MULTILINE)

_MALFORMED_OBS = (
    "No valid action found. Emit exactly one ```bash ... ``` block, or TASK_COMPLETE."
)


def parse_action(text: str) -> Action | None:
    """Fixed parser rho. Returns an Action or None if malformed."""
    if _DONE_RE.search(text):
        return Action(tool="submit", args={}, raw=text)
    m = _BASH_RE.search(text)
    if m:
        cmd = m.group(1).strip()
        return Action(tool="bash", args={"cmd": cmd}, raw=text)
    return None


def run_episode(
    task: TaskSpec,
    env: Environment,
    gamma: BaseContextManager,
    model: ModelClient,
    budget: BudgetCaps | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Trace:
    """Run one episode; return a full Trace (with per-step Gamma snapshots).

    This function is intentionally free of benchmark/model specifics — everything flows
    through the `Environment`, `ContextManager`, and `ModelClient` interfaces.
    """
    budget = budget or BudgetCaps()
    trace = Trace(task_id=task.task_id, benchmark=task.benchmark, model=model.model,
                  gamma_variant=gamma.name)

    gamma.reset(task, system_prompt)
    obs = env.reset()
    gamma.on_step(obs, action=None)  # seed with the initial observation

    total_tokens = 0
    reason = "budget"
    action: Action | None = None

    for t in range(budget.max_turns):
        messages = gamma.build_context(budget_tokens=budget.context_window)
        snap: GammaSnapshot = gamma.snapshot()

        result = model.chat(messages)
        total_tokens += result.total_tokens
        action = parse_action(result.text)

        if action is None:
            tool_result = _MALFORMED_OBS
            obs = Observation(text=tool_result)
        elif env.is_done(action):
            trace.steps.append(Step(t=t, observation=obs.text, gamma_snapshot=snap,
                                    action=action, tool_result="(submitted)"))
            reason = "submit"
            break
        else:
            obs = env.execute(action)
            tool_result = obs.text

        trace.steps.append(Step(t=t, observation=obs.text, gamma_snapshot=snap,
                                action=action, tool_result=tool_result))
        gamma.on_step(obs, action)

        if total_tokens >= budget.max_tokens:
            reason = "budget"
            break

    outcome: Outcome = env.verify()
    outcome.cost_tokens = total_tokens
    outcome.turns = len(trace.steps)
    if not outcome.terminated_reason:
        outcome.terminated_reason = reason
    trace.outcome = outcome
    env.close()
    return trace
