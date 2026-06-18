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

# Default cap on tool-calling steps per task (tau-bench's own default is 30).
MAX_NUM_STEPS = 30


def build_base_agent(env, model: str, provider: str, wiki: str | None = None):
    """
    Construct tau-bench's ToolCallingAgent from a (possibly gated) env.

    Real tau-bench signature:
        ToolCallingAgent(tools_info, wiki, model, provider, temperature=0.0)
    so tools_info/wiki must come from the env at run-time (not __init__).
    `wiki` can be overridden (used by Reflexion to inject a reflection).
    """
    try:
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent
    except ImportError:
        raise RuntimeError(
            "tau-bench not installed. Run: pip install git+https://github.com/sierra-research/tau-bench.git"
        )
    return ToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki if wiki is None else wiki,
        model=model,
        provider=provider,
    )


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

    def run(self, task, env, task_index: int = -1) -> AgentRunResult:
        gated = GatedEnv(env=env, critique_fn=None, env_name=self.env_name)
        base_agent = build_base_agent(env, self.model, self.model_provider)
        result = base_agent.solve(env=gated, task_index=task_index, max_num_steps=MAX_NUM_STEPS)
        return AgentRunResult(
            task_index=task_index,
            reward=float(result.reward),
            step_logs=gated.step_logs,
            metadata={
                "method": "vanilla",
                "model": self.model,
                "total_cost": getattr(result, "total_cost", None),
            },
        )


def _extract_reward(raw) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if hasattr(raw, "reward"):
        return float(raw.reward)
    if isinstance(raw, dict):
        return float(raw.get("reward", 0.0))
    return 0.0
