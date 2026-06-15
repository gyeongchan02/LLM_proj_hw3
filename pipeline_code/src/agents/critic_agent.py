"""
CriticGatingAgent — our primary method.

Wraps tau-bench's ToolCallingAgent with a GatedEnv whose critique_fn
calls a DIFFERENT-FAMILY model (Anthropic Claude) using the full
4-condition structured prompt.

Key differentiators vs SABER:
  1. Different model family (Claude vs GPT) → orthogonal errors
  2. 4-condition structured evaluation (Goal / State / Constraint / Policy)
  3. 4-way verdict including ask_user
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

from src.agents.gated_env import GatedEnv
from src.agents.vanilla import _extract_reward
from src.critic.critic import critique_ours
from src.critic.schemas import AgentRunResult

logger = logging.getLogger(__name__)


class CriticGatingAgent:
    """
    Our method: tau-bench base agent + Claude critic + GatedEnv.
    """

    def __init__(
        self,
        model: str,
        model_provider: str,
        critic_model: str = "gpt-5.4-mini",
        env_name: str = "retail",
        enable_rollback: bool = False,
        condition_ablation: Optional[list[str]] = None,
        **base_kwargs,
    ):
        self.model = model
        self.model_provider = model_provider
        self.critic_model = critic_model
        self.env_name = env_name
        self.enable_rollback = enable_rollback
        self.condition_ablation = condition_ablation
        self._base_agent = self._build_base_agent(**base_kwargs)

    def _build_base_agent(self, **kwargs):
        try:
            from tau_bench.agents.tool_calling_agent import ToolCallingAgent
            return ToolCallingAgent(
                model=self.model,
                model_provider=self.model_provider,
                **kwargs,
            )
        except ImportError:
            raise RuntimeError(
                "tau-bench not installed. Run: pip install -e external/tau-bench"
            )

    def _make_critique_fn(self, policy_text: str):
        critic_model = self.critic_model
        ablation = self.condition_ablation

        def critique_fn(
            tool_name, tool_args, goal, history_summary,
            policy_text=policy_text, task_index=None, **_
        ):
            return critique_ours(
                tool_name=tool_name,
                tool_args=tool_args,
                goal=goal,
                history_summary=history_summary,
                policy_text=policy_text,
                critic_model=critic_model,
                condition_ablation=ablation,
            )

        return critique_fn

    def run(self, task, env, task_index: int = -1) -> AgentRunResult:
        from src.data.action_taxonomy import get_policy_text
        policy_text = get_policy_text(self.env_name)
        critique_fn = self._make_critique_fn(policy_text)

        gated = GatedEnv(
            env=env,
            critique_fn=critique_fn,
            env_name=self.env_name,
            enable_rollback=self.enable_rollback,
        )
        gated.reset(task_index)

        raw = self._base_agent.run(task=task, env=gated)
        reward = _extract_reward(raw)

        logs = gated.step_logs
        return AgentRunResult(
            task_index=task_index,
            reward=reward,
            step_logs=logs,
            metadata={
                "method": "ours",
                "model": self.model,
                "critic_model": self.critic_model,   # gpt-5.4-mini
                "num_blocked": sum(1 for l in logs if l.decision == "block"),
                "num_revised": sum(1 for l in logs if l.decision == "revise"),
                "num_ask_user": sum(1 for l in logs if l.decision == "ask_user"),
                "condition_ablation": self.condition_ablation,
            },
        )
