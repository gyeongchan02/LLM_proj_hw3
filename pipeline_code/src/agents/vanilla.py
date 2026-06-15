"""
Vanilla baseline: tau-bench's ToolCallingAgent with no gating.

Rather than re-implementing the conversation loop, this class delegates
directly to tau-bench's agent and logs step data from the env wrapper.
"""

from __future__ import annotations

import logging

from src.agents.gated_env import GatedEnv
from src.critic.schemas import AgentRunResult

logger = logging.getLogger(__name__)


class VanillaAgent:
    """
    Wraps tau-bench's ToolCallingAgent unchanged (no critic).
    GatedEnv is used in passthrough mode (critique_fn=None) to collect step logs.
    """

    def __init__(
        self,
        model: str,
        model_provider: str,
        env_name: str = "retail",
        **kwargs,
    ):
        self.model = model
        self.model_provider = model_provider
        self.env_name = env_name
        self._base_agent = self._build_base_agent(**kwargs)

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

    def run(self, task, env, task_index: int = -1) -> AgentRunResult:
        gated = GatedEnv(env=env, critique_fn=None, env_name=self.env_name)
        gated.reset(task_index)
        raw = self._base_agent.run(task=task, env=gated)
        reward = _extract_reward(raw)
        return AgentRunResult(
            task_index=task_index,
            reward=reward,
            step_logs=gated.step_logs,
            metadata={"method": "vanilla", "model": self.model},
        )


def _extract_reward(raw) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if hasattr(raw, "reward"):
        return float(raw.reward)
    if isinstance(raw, dict):
        return float(raw.get("reward", 0.0))
    return 0.0
