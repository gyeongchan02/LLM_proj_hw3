"""
Critique functions — one per method variant.

critique_ours()   : gpt-5.4-mini (별도 모델, main agent와 다름)
critique_saber()  : gpt-5.4-nano (main agent와 동일 모델, 단순 프롬프트)
critique_oracle() : LLM 호출 없음, P2 레이블 파일 조회

All return Decision.  On parse error they fall back to ask_user.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

from src.critic.prompts import (
    CRITIC_SYSTEM_FULL,
    SABER_SYSTEM_PROMPT,
    build_critic_user_prompt,
    build_saber_user_prompt,
)
from src.critic.schemas import Decision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_decision(raw: str) -> Decision:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text.strip())
    verdict = data["verdict"]
    if verdict not in ("approve", "block", "revise", "ask_user"):
        raise ValueError(f"Unknown verdict: {verdict}")
    return Decision(
        verdict=verdict,
        reason=data.get("reason", ""),
        revised_args=data.get("revised_args"),
        question_to_user=data.get("question_to_user"),
        reversible=data.get("reversible"),
    )


def _fallback_decision(error: Exception) -> Decision:
    logger.warning(f"Critic parse error: {error}; falling back to ask_user")
    return Decision(
        verdict="ask_user",
        reason=f"Critic error: {error}",
        question_to_user="I need to verify this action. Can you confirm what you'd like me to do?",
        reversible=None,
    )


# ---------------------------------------------------------------------------
# Ours: gpt-5.4-mini critic (main agent와 다른 모델)
# ---------------------------------------------------------------------------

def critique_ours(
    tool_name: str,
    tool_args: dict[str, Any],
    goal: str,
    history_summary: str,
    policy_text: str,
    critic_model: str = "gpt-5.4-mini",
    condition_ablation: list[str] | None = None,
) -> Decision:
    """
    Full 4-condition critic using gpt-5.4-mini.
    Main agent runs on gpt-5.4-nano; using a stronger separate model here
    is the 'different-model' axis that distinguishes Ours from SABER.
    """
    user_prompt = build_critic_user_prompt(
        goal=goal,
        history_summary=history_summary,
        policy_text=policy_text,
        tool_name=tool_name,
        tool_args=tool_args,
        condition_ablation=condition_ablation,
    )
    try:
        response = litellm.completion(
            model=critic_model,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_FULL},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        return _parse_decision(response.choices[0].message.content or "")
    except Exception as e:
        return _fallback_decision(e)


# ---------------------------------------------------------------------------
# SABER: same model as main agent (gpt-5.4-nano), simple unstructured prompt
# ---------------------------------------------------------------------------

def critique_saber(
    tool_name: str,
    tool_args: dict[str, Any],
    history_summary: str,
    main_model: str,
) -> Decision:
    """
    SABER-style critic: same model as the main agent (gpt-5.4-nano), minimal prompt.
    """
    try:
        context = history_summary
        user_prompt = build_saber_user_prompt(tool_name, tool_args, context)
        response = litellm.completion(
            model=main_model,  # gpt-5.4-nano — same as main agent
            messages=[
                {"role": "system", "content": SABER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content or ""
        return _parse_decision(raw)
    except Exception as e:
        return _fallback_decision(e)


# ---------------------------------------------------------------------------
# Oracle: reads gold label from P2's JSONL label file
# ---------------------------------------------------------------------------

def _load_oracle_labels(label_file: str) -> dict[tuple[int, str], list[dict]]:
    """
    Load P2's perturbation labels.
    Keys: (task_index, tool_name) → list of entries (multiple perturbations per tool).

    JSONL format (from P2 / ACTION_PLAN Phase 1.2):
    {"task_id": "retail_012", "task_index": 12, "step": 7,
     "tool": "modify_pending_order_items",
     "args": {...}, "gold_decision": "revise", "reversible": true}
    """
    labels: dict[tuple[int, str], list[dict]] = {}
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = (entry["task_index"], entry["tool"])
            labels.setdefault(key, []).append(entry)
    return labels


def _find_label(candidates: list[dict], tool_args: dict) -> dict | None:
    """Return the candidate whose args exactly match tool_args, or None."""
    for c in candidates:
        if c.get("args") == tool_args:
            return c
    return None


_oracle_cache: dict[str, dict[tuple[int, str], list[dict]]] = {}


def critique_oracle(
    tool_name: str,
    tool_args: dict[str, Any],
    task_index: int,
    label_file: str,
) -> Decision:
    """
    Oracle critic: returns the gold verdict from P2's label file.
    No LLM call — represents the ceiling (perfect detector).
    """
    global _oracle_cache
    if label_file not in _oracle_cache:
        _oracle_cache[label_file] = _load_oracle_labels(label_file)

    labels = _oracle_cache[label_file]
    candidates = labels.get((task_index, tool_name), [])

    if not candidates:
        # No label for this (task, tool) pair → approve by default
        return Decision(verdict="approve", reason="No oracle label; defaulting to approve", reversible=True)

    entry = _find_label(candidates, tool_args)
    if entry is None:
        # Same (task, tool) exists in labels but args don't match any perturbation → approve
        return Decision(verdict="approve", reason="Args match no perturbation entry; defaulting to approve", reversible=True)

    verdict = entry["gold_decision"]
    return Decision(
        verdict=verdict,
        reason=f"Oracle label: {verdict}",
        revised_args=entry.get("revised_args"),
        question_to_user=entry.get("question_to_user"),
        reversible=entry.get("reversible"),
    )
