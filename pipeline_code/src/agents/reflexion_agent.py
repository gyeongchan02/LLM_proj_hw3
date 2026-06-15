"""
ReflexionAgent — Reflexion baseline.

Algorithm (Shinn et al. 2023, adapted for tau-bench):
  1. Run the full task with the base tool-calling agent (no gating).
  2. If reward == 0 and attempts < max_reflections:
     - Generate a "reflection" using the same model:
       "Here is what happened and what should be done differently."
     - Prepend reflection to the task instruction.
     - env.reset() and try again.
  3. Return the result of the last attempt.

Key properties (vs Ours):
  - Intervention is POST-HOC (after the task fails), not pre-action.
  - Uses the SAME model family as the main agent.
  - No rollback; retries from scratch.

These two axes (timing + model) are the exact ablation in README §2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.agents.gated_env import GatedEnv
from src.agents.vanilla import _extract_reward
from src.critic.schemas import AgentRunResult

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM = """\
You are a reflective module for a customer service AI.
Given the conversation history of a FAILED task, write a short reflection
(3-5 sentences) that:
  1. Identifies what went wrong (the key mistake).
  2. States what should be done differently.
  3. Notes any policy rule that was violated.
Be concrete. The reflection will be prepended to the next attempt.\
"""


class ReflexionAgent:
    """
    Post-hoc reflection loop wrapping tau-bench's ToolCallingAgent.
    No critic gating; reflects after task failure.
    """

    def __init__(
        self,
        model: str,
        model_provider: str,
        max_reflections: int = 3,
        env_name: str = "retail",
        **base_kwargs,
    ):
        self.model = model
        self.model_provider = model_provider
        self.max_reflections = max_reflections
        self.env_name = env_name
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

    def run(self, task, env, task_index: int = -1) -> AgentRunResult:
        reflection: str | None = None
        result = None

        for attempt in range(self.max_reflections + 1):
            # Apply reflection to task if available
            augmented_task = self._augment_task(task, reflection)

            gated = GatedEnv(env=env, critique_fn=None, env_name=self.env_name)
            gated.reset(task_index)

            raw = self._base_agent.run(task=augmented_task, env=gated)
            reward = _extract_reward(raw)
            logs = gated.step_logs

            result = AgentRunResult(
                task_index=task_index,
                reward=reward,
                step_logs=logs,
                metadata={
                    "method": "reflexion",
                    "model": self.model,
                    "attempt": attempt,
                    "reflection": reflection,
                },
            )

            if reward > 0 or attempt >= self.max_reflections:
                break

            # Generate reflection for next attempt
            messages = _extract_messages(raw)
            reflection = self._generate_reflection(messages)
            logger.info(f"Reflexion attempt {attempt + 1} failed; reflecting...")

        return result

    def _augment_task(self, task, reflection: str | None):
        if reflection is None:
            return task
        prefix = f"[REFLECTION FROM PREVIOUS ATTEMPT]\n{reflection}\n\n"
        if hasattr(task, "instruction"):
            try:
                augmented = object.__new__(type(task))
                augmented.__dict__.update(task.__dict__)
                augmented.instruction = prefix + task.instruction
                return augmented
            except Exception:
                pass
        # If task is a plain string
        return prefix + str(task)

    def _generate_reflection(self, messages: list[dict]) -> str:
        try:
            import litellm
            history_text = _messages_to_text(messages, max_chars=3000)
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": REFLECTION_SYSTEM},
                    {"role": "user", "content": f"Conversation history:\n{history_text}\n\nWrite your reflection."},
                ],
                temperature=0,
                max_tokens=300,
            )
            return response.choices[0].message.content or "No reflection generated."
        except Exception as e:
            logger.warning(f"Reflection generation failed: {e}")
            return "Unable to generate reflection."


def _extract_messages(raw) -> list[dict]:
    if hasattr(raw, "messages"):
        return raw.messages or []
    if isinstance(raw, dict):
        return raw.get("messages", [])
    return []


def _messages_to_text(messages: list[dict], max_chars: int = 3000) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = str(content)
        if content:
            lines.append(f"[{role.upper()}] {str(content)[:300]}")
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text
