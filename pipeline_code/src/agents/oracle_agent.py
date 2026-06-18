"""
OracleGatingAgent — ceiling baseline.

Uses P2's gold labels instead of an LLM to decide each verdict.
Represents "what if the critic were perfect?"

The oracle uses the same GatedEnv infrastructure as Ours/SABER so any
difference in final score vs Ours is attributable only to critic accuracy,
not to the gating/recovery mechanism.

IMPORTANT: The oracle reads from P2's label file.
The label file path is configured via:
  - --oracle-labels CLI flag  (run_experiment.sh)
  - configs/experiment.yaml   (oracle.label_file)
"""

from __future__ import annotations

import logging

from src.agents.gated_env import GatedEnv
from src.agents.vanilla import build_base_agent, MAX_NUM_STEPS
from src.critic.critic import critique_oracle
from src.critic.schemas import AgentRunResult

logger = logging.getLogger(__name__)


class OracleGatingAgent:
    """
    Oracle critic: gold-label verdicts from P2's perturbation dataset.
    """

    def __init__(
        self,
        model: str,
        model_provider: str,
        label_file: str,                    # path to P2's JSONL label file
        env_name: str = "retail",
        **base_kwargs,
    ):
        self.model = model
        self.model_provider = model_provider
        self.label_file = label_file
        self.env_name = env_name

    def _make_critique_fn(self):
        label_file = self.label_file

        def critique_fn(
            tool_name, tool_args, goal, history_summary,
            policy_text=None, task_index=-1, **_
        ):
            return critique_oracle(
                tool_name=tool_name,
                tool_args=tool_args,
                task_index=task_index,
                label_file=label_file,
            )

        return critique_fn

    def run(self, task, env, task_index: int = -1) -> AgentRunResult:
        critique_fn = self._make_critique_fn()

        gated = GatedEnv(
            env=env,
            critique_fn=critique_fn,
            env_name=self.env_name,
        )
        base_agent = build_base_agent(env, self.model, self.model_provider)
        result = base_agent.solve(env=gated, task_index=task_index, max_num_steps=MAX_NUM_STEPS)

        logs = gated.step_logs
        return AgentRunResult(
            task_index=task_index,
            reward=float(result.reward),
            step_logs=logs,
            metadata={
                "method": "oracle",
                "model": self.model,
                "label_file": self.label_file,
                "total_cost": getattr(result, "total_cost", None),
                "num_blocked": sum(1 for l in logs if l.decision == "block"),
                "num_revised": sum(1 for l in logs if l.decision == "revise"),
                "num_ask_user": sum(1 for l in logs if l.decision == "ask_user"),
            },
        )
