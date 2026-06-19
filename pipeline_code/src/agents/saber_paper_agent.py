"""
SaberPaperAgent — faithful reproduction of SABER (Cuadron et al. 2025), core path.

Implements the two gain-driving mechanisms from the paper (§4, Table 3):

  1. Mutation-Gated USER Verification
     Only MUTATING actions are gated. The auxiliary reformulates the tool call
     into a plain-language confirmation question and asks the *user simulator*
     (env.user.step) to confirm. The user's reply is injected into the
     trajectory; the MAIN model then re-issues (execute) or revises. Crucially,
     "the next post-feedback action is executed directly" (§4.1) — this prevents
     the prompt-locking / stalling that an autonomous block-based critic causes.

  2. Targeted Reflection
     A one-line, high-salience policy reminder is injected at the point of
     mutation (ReAct-style fallback, §4.2), to reduce miscalibrated tool calls.

NOT implemented (by scope decision): Mechanism 3, block-based context cleaning,
which needs a custom agent loop + per-turn embedding retrieval. Non-mutating
actions bypass the gate entirely (paper §4.1).

Default pairing is same-model (main == auxiliary), matching SABER's default.

Difference vs the team's original `saber` baseline: that one is an autonomous
4-way LLM critic that never reaches the user. This one routes confirmation to
the user simulator and lets the user decide — i.e. SABER's actual core.
"""

from __future__ import annotations

import logging

from src.agents.gated_env import GatedEnv, _stable_key
from src.agents.vanilla import build_base_agent, MAX_NUM_STEPS
from src.critic.critic import saber_verify, saber_user_confirmed
from src.critic.schemas import AgentRunResult, StepLog
from src.data.action_taxonomy import get_policy_text, is_reversible

logger = logging.getLogger(__name__)


class SaberGatedEnv(GatedEnv):
    """
    GatedEnv variant implementing SABER's mutation-gated USER verification.

    Paper-faithful gate (Cuadron et al. §4.2): for EVERY candidate action, the
    AUXILIARY MODEL decides whether it is mutating (we do NOT use the deterministic
    is_mutating taxonomy here — that's what `ours`/oracle use). If the aux says
    non-mutating → bypass. If mutating → reformulate + ask the LIVE user simulator:
      · user confirms → execute the action now.
      · user rejects  → don't execute; hand back for revision.
    Already-verified actions execute directly (anti-stall, §4.1).

    Operational note: the paper has the *main model* re-issue the action after the
    user's feedback; we execute synchronously on confirm instead (small main model
    is unreliable at re-issuing; tau-bench appends "###STOP###" to confirmation turns).
    Outcome is the same; "###STOP###" is stripped so a confirmation can't end the episode.
    """

    def __init__(self, env, env_name: str, aux_model: str, policy_text: str):
        super().__init__(env=env, critique_fn=None, env_name=env_name)
        self._aux_model = aux_model
        self._policy_text = policy_text
        self._verified_keys: set[str] = set()   # actions already gated this task
        self.num_verifications = 0
        self.num_executed_after_confirm = 0
        self.num_user_rejected = 0
        self.num_aux_calls = 0                   # aux mutating-classification calls

    def reset(self, task_index=None):
        self._verified_keys = set()
        return super().reset(task_index)

    def step(self, action):
        from tau_bench.types import EnvResponse

        name, kwargs = self._parse_action(action)
        self._step_num += 1
        rev = is_reversible(name, self._env_name)   # taxonomy hint, for logging only

        # ── Already gated once → execute directly (anti-stall; skip aux + user) ─
        key = f"{name}:{_stable_key(kwargs)}"
        if key in self._verified_keys:
            result = self._env.step(action)
            self._last_info = getattr(result, "info", self._last_info)
            self.num_executed_after_confirm += 1
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=kwargs,
                is_mutating=True, reversible=rev, decision="execute_post_confirm", executed=True,
            ))
            return result

        # ── Paper-faithful: the AUX MODEL classifies mutating (+ reformulates) ─
        mutating, reminder, confirmation = saber_verify(
            tool_name=name, tool_args=kwargs, goal=self._goal,
            history_summary=self._history_summary(),
            policy_text=self._policy_text, aux_model=self._aux_model,
        )
        self.num_aux_calls += 1

        # ── Aux says non-mutating → bypass the gate ─────────────────────────
        if not mutating:
            result = self._env.step(action)
            self._last_info = getattr(result, "info", self._last_info)
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=kwargs,
                is_mutating=False, reversible=None, decision=None, executed=True,
            ))
            return result

        # ── Mutating (per aux) → targeted reflection + USER verification ────
        self._verified_keys.add(key)
        # Route confirmation to the LIVE user simulator; strip the simulator's
        # conversation-ending marker so a confirmation never terminates the episode.
        raw_reply = self._env.user.step(confirmation) or ""
        user_reply = raw_reply.replace("###STOP###", "").strip()
        self.num_verifications += 1

        confirmed = saber_user_confirmed(user_reply, self._aux_model)

        if confirmed:
            # User approved → execute the action now.
            result = self._env.step(action)
            self._last_info = getattr(result, "info", self._last_info)
            self.num_executed_after_confirm += 1
            self._step_logs.append(StepLog(
                step=self._step_num, tool=name, args=kwargs,
                is_mutating=True, reversible=rev,
                decision="approve_after_user_confirm", executed=True,
            ))
            new_obs = (
                f"[SABER REMINDER] {reminder}\n"
                f"[SABER] You confirmed with the customer (\"{user_reply}\") before "
                f"executing '{name}'.\n{getattr(result, 'observation', '')}"
            )
            try:
                return result.model_copy(update={"observation": new_obs})
            except Exception:
                return result

        # User did NOT confirm → do not execute; hand back for revision.
        self.num_user_rejected += 1
        self._step_logs.append(StepLog(
            step=self._step_num, tool=name, args=kwargs,
            is_mutating=True, reversible=rev,
            decision="reject_after_user", executed=False,
        ))
        obs = (
            f"[SABER REMINDER] {reminder}\n"
            f"[SABER USER VERIFICATION] Before executing '{name}' with {kwargs}, I asked the "
            f"customer to confirm. They did NOT confirm — they said: \"{user_reply}\"\n"
            "Do not execute that action. Revise it to match what the customer actually wants, "
            "or ask them a clarifying question."
        )
        return EnvResponse(observation=obs, reward=0.0, done=False, info=self._last_info)


class SaberPaperAgent:
    """Faithful SABER (core path). Same model for main + auxiliary by default."""

    def __init__(
        self,
        model: str,
        model_provider: str,
        env_name: str = "retail",
        aux_model: str | None = None,
        **base_kwargs,
    ):
        self.model = model
        self.model_provider = model_provider
        self.env_name = env_name
        self.aux_model = aux_model or model   # SABER default: aux == main

    def run(self, task, env, task_index: int = -1) -> AgentRunResult:
        gated = SaberGatedEnv(
            env=env, env_name=self.env_name,
            aux_model=self.aux_model, policy_text=get_policy_text(self.env_name),
        )
        base_agent = build_base_agent(env, self.model, self.model_provider)
        result = base_agent.solve(env=gated, task_index=task_index, max_num_steps=MAX_NUM_STEPS)

        logs = gated.step_logs
        return AgentRunResult(
            task_index=task_index,
            reward=float(result.reward),
            step_logs=logs,
            metadata={
                "method": "saber",
                "model": self.model,
                "aux_model": self.aux_model,
                "total_cost": getattr(result, "total_cost", None),
                "num_verifications": gated.num_verifications,
                "num_executed_after_confirm": gated.num_executed_after_confirm,
                "num_user_rejected": gated.num_user_rejected,
                "num_aux_calls": gated.num_aux_calls,
            },
        )
