"""
GatedEnv — wraps a tau-bench Env to intercept mutating tool calls.

All four critic variants (Ours, SABER, Oracle, Vanilla) go through this
wrapper.  The difference between methods is only the critique_fn passed in.
Vanilla passes critique_fn=None to bypass gating entirely.

Tau-bench compatibility note
────────────────────────────
We assume env.step(action) where action satisfies one of:
  (a) has .name (str) and .kwargs (dict)     ← tau_bench.types.Action
  (b) is a dict {"name": ..., "kwargs": ...} ← plain dict
  (c) is a dict {"name": ..., "arguments": ...} ← OpenAI tool-call dict

env.step returns (observation: str, reward: float, done: bool, info: dict).
If tau-bench returns only observation (not a tuple), adapt _wrap_result().

If the actual method names or formats differ, edit _parse_action() and
_make_action() below — all env interactions are funnelled through those two.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from src.critic.schemas import Decision, StepLog
from src.data.action_taxonomy import get_policy_text, is_mutating, is_reversible

logger = logging.getLogger(__name__)

MAX_RETRY_SAME_ACTION = 2   # prevent infinite revise loops


class GatedEnv:
    """
    Wraps a tau-bench Env.
    - Non-mutating tools → pass through unchanged.
    - Mutating tools     → call critique_fn, act on verdict.

    If critique_fn is None, all actions pass through (Vanilla mode).
    """

    def __init__(
        self,
        env,
        critique_fn: Optional[Callable[..., Decision]],
        env_name: str = "retail",
        enable_rollback: bool = False,
        ask_user_to_sim: bool = False,
        aux_model: Optional[str] = None,
    ):
        self._env = env
        self._critique = critique_fn
        self._env_name = env_name
        self._enable_rollback = enable_rollback
        # When True, an `ask_user` verdict routes the critic's question to the LIVE
        # user simulator (env.user.step) and acts on the reply — used by "ours" as
        # the upgraded-SABER path. approve/revise/block stay autonomous.
        self._ask_user_to_sim = ask_user_to_sim
        self._aux_model = aux_model

        # Per-task state (reset on each env.reset())
        self._goal: str = ""
        self._task_index: int = -1
        self._step_logs: list[StepLog] = []
        self._retry_counts: dict[str, int] = {}
        self._asked_keys: set[str] = set()   # actions already routed to the user
        self._step_num: int = 0
        self._last_info = None   # last real EnvInfo (reused for synthetic responses)

    # ── Public interface ─────────────────────────────────────────────────────

    def reset(self, task_index: int | None = None):
        # tau-bench's Agent.solve calls env.reset(task_index=...), so this is the
        # authoritative reset. Returns an EnvResetResponse (observation/info).
        if task_index is not None:
            self._task_index = task_index
        self._step_logs = []
        self._retry_counts = {}
        self._asked_keys = set()
        self._step_num = 0
        res = self._env.reset(task_index=task_index)
        self._goal = str(getattr(res, "observation", "") or "")
        self._last_info = getattr(res, "info", None)
        return res

    # Forward the attributes tau-bench's ToolCallingAgent reads off the env.
    @property
    def tools_info(self):
        return self._env.tools_info

    @property
    def wiki(self):
        return self._env.wiki

    @property
    def tools(self):
        return getattr(self._env, "tools_info", None)

    @property
    def step_logs(self) -> list[StepLog]:
        return list(self._step_logs)

    @property
    def task_index(self) -> int:
        return self._task_index

    def step(self, action) -> tuple:
        name, kwargs = self._parse_action(action)
        self._step_num += 1

        # ── Non-mutating: pass straight through ─────────────────────────────
        if not is_mutating(name, self._env_name) or self._critique is None:
            result = self._env.step(action)
            self._last_info = getattr(result, "info", self._last_info)
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=kwargs,
                is_mutating=is_mutating(name, self._env_name),
                reversible=None, decision=None, executed=True,
                observation=str(getattr(result, "observation", ""))[:800],
            ))
            return result

        # ── Mutating: consult critic ─────────────────────────────────────────
        decision: Decision = self._critique(
            tool_name=name,
            tool_args=kwargs,
            goal=self._goal,
            history_summary=self._history_summary(),
            policy_text=get_policy_text(self._env_name),
            task_index=self._task_index,  # needed by oracle; ignored by others
        )

        verdict = decision.verdict
        reversible_hint = (
            decision.reversible
            if decision.reversible is not None
            else is_reversible(name, self._env_name)
        )

        # ── approve ──────────────────────────────────────────────────────────
        if verdict == "approve":
            result = self._env.step(action)
            self._last_info = getattr(result, "info", self._last_info)
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=kwargs,
                is_mutating=True, reversible=reversible_hint,
                decision="approve", executed=True,
                observation=str(getattr(result, "observation", ""))[:800],
            ))
            return result

        # ── revise ───────────────────────────────────────────────────────────
        if verdict == "revise":
            action_key = f"{name}:{_stable_key(kwargs)}"
            count = self._retry_counts.get(action_key, 0) + 1
            self._retry_counts[action_key] = count

            if count > MAX_RETRY_SAME_ACTION:
                logger.warning(f"Max retries for {name}; escalating to ask_user")
                self._step_logs.append(StepLog(
                    step=self._step_num, tool=name, args=kwargs,
                    is_mutating=True, reversible=reversible_hint,
                    decision="ask_user", executed=False,
                ))
                return self._blocked_result(
                    name,
                    f"Max retries exceeded for {name}. "
                    "Please clarify what you need.",
                )

            revised = decision.revised_args or kwargs
            revised_action = self._make_action(name, revised, action)
            result = self._env.step(revised_action)
            self._last_info = getattr(result, "info", self._last_info)
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=revised,
                is_mutating=True, reversible=reversible_hint,
                decision="revise", executed=True,
                observation=str(getattr(result, "observation", ""))[:800],
            ))
            return self._append_feedback(
                result,
                f"[CRITIC-REVISE] {decision.reason}",
            )

        # ── block ────────────────────────────────────────────────────────────
        if verdict == "block":
            # Fix #1: ours never blocks autonomously — route the concern to the
            # customer, who decides (confirm → execute, reject → revise).
            if self._ask_user_to_sim:
                q = (f"I was about to run '{name}' with {kwargs}, but I have a concern: "
                     f"{decision.reason} Should I proceed?")
                return self._route_to_user(action, name, kwargs, q, reversible_hint, "block")
            # default (autonomous, e.g. oracle): synthetic block message to the agent
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=kwargs,
                is_mutating=True, reversible=reversible_hint,
                decision="block", executed=False,
            ))
            return self._blocked_result(
                name,
                f"[CRITIC-BLOCK] {decision.reason}. "
                "Reconsider your approach.",
            )

        # ── ask_user ─────────────────────────────────────────────────────────
        if verdict == "ask_user":
            question = decision.question_to_user or "Can you clarify your request?"
            if self._ask_user_to_sim:
                return self._route_to_user(action, name, kwargs, question, reversible_hint, "ask_user")
            # default (autonomous, e.g. oracle): synthetic message to the agent
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=kwargs,
                is_mutating=True, reversible=reversible_hint,
                decision="ask_user", executed=False,
            ))
            return self._blocked_result(
                name,
                f"[CRITIC-ASK] {question}",
            )

        # Fallback (should not reach here)
        return self._env.step(action)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _parse_action(self, action) -> tuple[str, dict[str, Any]]:
        if hasattr(action, "name") and hasattr(action, "kwargs"):
            return action.name, action.kwargs or {}
        if isinstance(action, dict):
            name = action.get("name", "")
            kwargs = action.get("kwargs", action.get("arguments", {}))
            if isinstance(kwargs, str):
                import json
                kwargs = json.loads(kwargs)
            return name, kwargs or {}
        return str(action), {}

    def _make_action(self, name: str, kwargs: dict, original_action):
        """Create a revised action with the same type as the original."""
        if hasattr(original_action, "name") and hasattr(original_action, "kwargs"):
            try:
                from tau_bench.types import Action
                return Action(name=name, kwargs=kwargs)
            except ImportError:
                pass
            # Fallback: shallow-copy and replace kwargs
            obj = object.__new__(type(original_action))
            obj.__dict__.update(original_action.__dict__)
            obj.kwargs = kwargs
            return obj
        return {"name": name, "kwargs": kwargs}

    def _blocked_result(self, tool_name: str, message: str):
        """Return a synthetic EnvResponse for blocked/ask_user actions.

        The action is NOT executed, so reward=0.0 and done=False. We reuse the
        last real EnvInfo so that tau-bench's solve() can call info.model_dump().
        """
        from tau_bench.types import EnvResponse
        obs = (
            f"Tool '{tool_name}' was NOT executed.\n"
            f"{message}\n"
            "Please adjust your approach and try again."
        )
        return EnvResponse(
            observation=obs, reward=0.0, done=False, info=self._last_info,
        )

    def _append_feedback(self, result, feedback: str):
        """Append critic feedback to the observation of a real EnvResponse."""
        obs = f"{getattr(result, 'observation', '')}\n{feedback}"
        try:
            return result.model_copy(update={"observation": obs})
        except Exception:
            # Fallback for unexpected response shapes
            from tau_bench.types import EnvResponse
            return EnvResponse(
                observation=obs,
                reward=getattr(result, "reward", 0.0),
                done=getattr(result, "done", False),
                info=getattr(result, "info", self._last_info),
            )

    def _route_to_user(self, action, name, kwargs, question, reversible_hint, label):
        """Fix #1 (ours): route a critic concern (block OR ask_user) to the LIVE
        user simulator — the customer arbitrates (confirm → execute, reject → revise).
        The critic never decides block autonomously. `label` records the original
        verdict ('block' or 'ask_user'); `executed` records the post-user outcome."""
        from src.critic.critic import saber_user_confirmed
        key = f"{name}:{_stable_key(kwargs)}"
        # Anti-stall: an action already routed once executes directly on re-proposal.
        if key in self._asked_keys:
            result = self._env.step(action)
            self._last_info = getattr(result, "info", self._last_info)
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=kwargs,
                is_mutating=True, reversible=reversible_hint, decision=label, executed=True,
            ))
            return result
        self._asked_keys.add(key)

        raw = self._env.user.step(question) or ""
        user_reply = raw.replace("###STOP###", "").strip()
        if saber_user_confirmed(user_reply, self._aux_model):
            result = self._env.step(action)
            self._last_info = getattr(result, "info", self._last_info)
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=kwargs,
                is_mutating=True, reversible=reversible_hint, decision=label, executed=True,
            ))
            return self._append_feedback(result, f"[CRITIC→USER confirmed] {user_reply}")
        # customer did not confirm → do not execute; hand back for revision
        self._step_logs.append(StepLog(
            step=self._step_num, tool=name, args=kwargs,
            is_mutating=True, reversible=reversible_hint, decision=label, executed=False,
        ))
        return self._blocked_result(
            name,
            f"[CRITIC→USER] The customer did NOT confirm: \"{user_reply}\". "
            "Revise the action to match what they want.",
        )

    def _history_summary(self, n: int = 6) -> str:
        if not self._step_logs:
            return "No previous tool calls."
        lines = []
        for log in self._step_logs[-n:]:
            status = "OK" if log.executed else f"NOT_EXECUTED({log.decision})"
            lines.append(f"  step {log.step}: {log.tool}({log.args}) → {status}")
        return "\n".join(lines)

    def _full_transcript(self) -> str:
        """FULL conversation so far for the critic: every prior action WITH its
        observation (customer replies + tool/DB results). Unlike _history_summary
        (action-only, last-6), this lets the critic actually verify the action."""
        if not self._step_logs:
            return "No prior conversation."
        lines = []
        for log in self._step_logs:
            if log.tool == "respond":
                content = (log.args or {}).get("content", "")
                lines.append(f"Agent → customer: {str(content)[:800]}")
                if log.observation:
                    lines.append(f"  Customer replied: {log.observation}")
            else:
                note = "" if log.executed else f"  [NOT executed: {log.decision}]"
                lines.append(f"Agent called {log.tool}({log.args}){note}")
                if log.observation:
                    lines.append(f"  → Result: {log.observation}")
        return "\n".join(lines)


def _stable_key(d: dict) -> str:
    return str(sorted(d.items()))
