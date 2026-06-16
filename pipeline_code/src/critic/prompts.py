"""
Prompt templates for the critic.

Two variants:
  - FULL (Ours): 4-condition structured evaluation, used with a different-family model.
  - SABER: minimal single-pass prompt, used with the same model as the main agent.

Ablation note (P3): to ablate individual conditions remove the corresponding
section from CRITIC_SYSTEM_FULL and set condition_ablation= in build_critic_user_prompt().
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# Ours: 4-condition critic (different model, structured)
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_FULL = """\
You are a Critic Agent reviewing proposed tool calls for a customer service AI.
Decide whether the proposed action should proceed.

Evaluate along EXACTLY these 4 dimensions (omit none):

1. GOAL    — Does this action advance the customer's stated request?
2. STATE   — Is this action consistent with what we know about the current DB state
             (orders, user profile, product info gathered so far)?
3. CONSTRAINT — Are the tool name and all argument values technically valid
             (correct types, required fields present, enum values in range)?
4. POLICY  — Does this action comply with the domain business rules listed below?

Verdict options (choose exactly one):
  "approve"   — All 4 conditions satisfied; execute as-is.
  "revise"    — Tool/intent and target are correct, but specific argument VALUES
                need fixing. Provide corrected args in revised_args.
  "block"     — Action clearly violates policy or the current state, or is
                otherwise wrong; do NOT execute. Use for any clear violation
                REGARDLESS of reversibility — refuse it; never ask the user to
                approve a violation.
  "ask_user"  — Insufficient information to decide, OR the action is VALID but
                irreversible / high-stakes and needs explicit user confirmation
                before proceeding.

Rules:
  - Prefer "revise" over "block" when only argument values are wrong but the
    intent and target are right.
  - "block" a clear policy/state violation even if the tool is irreversible
    (you refuse it, so reversibility is irrelevant).
  - "ask_user" only when you are genuinely uncertain / info is missing, or a
    *valid* action is irreversible and needs sign-off.
  - Default to "ask_user" if you cannot confidently choose another verdict.
  - NEVER output anything outside the JSON object.

Return ONLY a JSON object with this exact schema:
{
  "verdict": "approve" | "block" | "revise" | "ask_user",
  "reason": "<one concise sentence>",
  "revised_args": { ... } | null,
  "question_to_user": "<question string>" | null,
  "reversible": true | false
}\
"""


def build_critic_user_prompt(
    goal: str,
    history_summary: str,
    policy_text: str,
    tool_name: str,
    tool_args: dict,
    condition_ablation: list[str] | None = None,
) -> str:
    """
    Build the user-turn prompt for the full 4-condition critic.

    condition_ablation: list of conditions to OMIT, e.g. ["POLICY"] to ablate
    the policy condition.  Used by P3 for ablation experiments.
    """
    skip = set(condition_ablation or [])

    sections = []

    sections.append(f"CUSTOMER GOAL:\n{goal}")
    sections.append(f"RECENT ACTION HISTORY:\n{history_summary}")

    if "POLICY" not in skip:
        sections.append(f"DOMAIN POLICY:\n{policy_text}")
    else:
        sections.append("DOMAIN POLICY:\n[ABLATED — ignore policy dimension]")

    sections.append(
        f"PROPOSED ACTION:\n"
        f"  tool: {tool_name}\n"
        f"  args: {json.dumps(tool_args, ensure_ascii=False, indent=2)}"
    )

    if skip:
        skipped = ", ".join(sorted(skip))
        sections.append(f"[ABLATION: ignore dimension(s): {skipped}]")

    sections.append("Evaluate and return your verdict as JSON.")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# SABER baseline: simple unstructured single-pass prompt, same model
# ---------------------------------------------------------------------------

SABER_SYSTEM_PROMPT = """\
You are a reflection module for a customer service AI.
Check whether the proposed tool call is correct before it is executed.
Return ONLY a JSON object:
{
  "verdict": "approve" | "block" | "revise" | "ask_user",
  "reason": "<one sentence>",
  "revised_args": { ... } | null,
  "question_to_user": "<question>" | null,
  "reversible": true | false
}\
"""


def build_saber_user_prompt(tool_name: str, tool_args: dict, context: str) -> str:
    return (
        f"Context so far:\n{context}\n\n"
        f"Proposed action: {tool_name}({json.dumps(tool_args)})\n\n"
        "Is this action correct? Return your verdict as JSON."
    )
