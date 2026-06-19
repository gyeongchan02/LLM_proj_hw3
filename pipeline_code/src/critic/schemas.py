from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# Four-way verdict returned by the critic.
Verdict = Literal["approve", "block", "revise", "ask_user"]


@dataclass
class Decision:
    """Single decision produced by the critic for one proposed action."""
    verdict: Verdict
    reason: str
    revised_args: Optional[dict[str, Any]] = None     # populated when verdict == "revise"
    question_to_user: Optional[str] = None             # populated when verdict == "ask_user"
    reversible: Optional[bool] = None                  # critic's estimate of action reversibility
    rollback_to_step: Optional[int] = None             # stretch: step to roll back to


@dataclass
class StepLog:
    """Execution record for one env.step call (logged by GatedEnv)."""
    step: int
    tool: str
    args: dict[str, Any]
    is_mutating: bool
    reversible: Optional[bool]
    decision: Optional[str]   # verdict string, or None for non-mutating
    executed: bool            # whether env.step was actually called
    rolled_back: bool = False
    observation: Optional[str] = None   # result / customer reply (for full-trajectory critic context)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "tool": self.tool,
            "args": self.args,
            "is_mutating": self.is_mutating,
            "reversible": self.reversible,
            "decision": self.decision,
            "executed": self.executed,
            "rolled_back": self.rolled_back,
        }


@dataclass
class AgentRunResult:
    """Unified result object returned by every agent.run()."""
    task_index: int
    reward: float
    step_logs: list[StepLog] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_index": self.task_index,
            "reward": self.reward,
            "step_logs": [s.to_dict() for s in self.step_logs],
            "metadata": self.metadata,
        }
