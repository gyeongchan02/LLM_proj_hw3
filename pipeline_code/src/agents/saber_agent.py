"""
SABERGatingAgent — SABER baseline.

Same architecture as CriticGatingAgent but:
  - SAME model family as the main agent (no cross-family separation)
  - Simple unstructured prompting (no 4-condition structure)
  - Still 4-way verdict output (for fair gating comparison)
  - No rollback

This isolates the contribution of "different model family" and
"structured prompt" when compared with Ours.
"""

from __future__ import annotations

import logging

from src.agents.gated_env import GatedEnv
from src.agents.vanilla import build_base_agent, MAX_NUM_STEPS
from src.critic.critic import critique_saber
from src.critic.schemas import AgentRunResult

logger = logging.getLogger(__name__)


class SABERGatingAgent:
    """
    SABER-style: same model as main agent, unstructured prompt, no rollback.
    """

    def __init__(
        self,
        model: str,
        model_provider: str,
        env_name: str = "retail",
        **base_kwargs,
    ):
        self.model = model
        self.model_provider = model_provider
        self.env_name = env_name

    def _make_critique_fn(self):
        model = self.model

        def critique_fn(
            tool_name, tool_args, goal, history_summary,
            policy_text=None, task_index=None, **_
        ):
            return critique_saber(
                tool_name=tool_name,
                tool_args=tool_args,
                history_summary=history_summary,
                main_model=model,
            )

        return critique_fn

    def run(self, task, env, task_index: int = -1) -> AgentRunResult:
        critique_fn = self._make_critique_fn()

        gated = GatedEnv(
            env=env,
            critique_fn=critique_fn,
            env_name=self.env_name,
            enable_rollback=False,
        )
        base_agent = build_base_agent(env, self.model, self.model_provider)
        result = base_agent.solve(env=gated, task_index=task_index, max_num_steps=MAX_NUM_STEPS)

        logs = gated.step_logs
        return AgentRunResult(
            task_index=task_index,
            reward=float(result.reward),
            step_logs=logs,
            metadata={
                "method": "saber_old",
                "model": self.model,
                "critic_model": self.model,
                "total_cost": getattr(result, "total_cost", None),
                "num_blocked": sum(1 for l in logs if l.decision == "block"),
                "num_revised": sum(1 for l in logs if l.decision == "revise"),
                "num_ask_user": sum(1 for l in logs if l.decision == "ask_user"),
            },
        )
