"""
SaberBlockAgent — full 3-mechanism SABER (Cuadron et al. 2025): "SABER+Block".

Adds Mechanism 3 (Block-based Context Cleaning, §4.1/§4.2) on top of the faithful
SABER core (Mechanism 1 user-gated verification + Mechanism 2 targeted reflection,
both provided by SaberGatedEnv).

Because mechanism 3 must control what the MAIN agent sees each turn, we cannot use
tau-bench's ToolCallingAgent.solve (which always keeps the full trajectory). We
re-implement that loop here and, each turn, build a CLEANED context:

  effective context = system(wiki) + goal + [top-N relevant older blocks, summarized]
                      + [last RECENT_ROUNDS rounds, raw]

A "block" = one action↔result round. Older rounds are summarized (auxiliary LLM,
one line each, cached) and embedded (text-embedding-3-small). For the current step
we retrieve the N most relevant older blocks by cosine similarity to the latest
observation (paper: "N most relevant blocks by embedding similarity to the latest
user query"), N capped at 16 (paper Table 3). Recent rounds stay raw to preserve
immediate tool-call coherence (and valid OpenAI tool_call/tool pairing).

Faithfulness notes: blocks are segmented per action-result round (the paper does not
fully specify the boundary rule); retrieval query = latest observation. Embeddings +
summaries use the MAIN key (agent-side), the user-sim uses its own key.
"""

from __future__ import annotations

import logging
import math

from src.agents.saber_paper_agent import SaberGatedEnv
from src.critic.schemas import AgentRunResult
from src.data.action_taxonomy import get_policy_text

logger = logging.getLogger(__name__)

MAX_NUM_STEPS = 30
RECENT_ROUNDS = 3          # keep this many most-recent rounds raw
N_BLOCKS = 16              # retrieve top-N older blocks (paper cap)
EMBED_MODEL = "text-embedding-3-small"


def _cos(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class SaberBlockAgent:
    def __init__(
        self,
        model: str,
        model_provider: str,
        env_name: str = "retail",
        aux_model: str | None = None,
        n_blocks: int = N_BLOCKS,
        recent_rounds: int = RECENT_ROUNDS,
        **kw,
    ):
        self.model = model
        self.model_provider = model_provider
        self.env_name = env_name
        self.aux_model = aux_model or model
        self.n_blocks = n_blocks
        self.recent_rounds = recent_rounds

    # ── Mechanism 3 helpers ───────────────────────────────────────────────────
    def _summarize(self, text: str) -> str:
        import litellm
        try:
            r = litellm.completion(
                model=self.aux_model, custom_llm_provider=self.model_provider,
                messages=[
                    {"role": "system", "content":
                     "Summarize this customer-service exchange in ONE concise sentence. "
                     "Keep concrete order IDs, item IDs, amounts, and any decision/outcome."},
                    {"role": "user", "content": text[:2000]},
                ],
                temperature=0, max_tokens=80,
            )
            return (r.choices[0].message.content or "").strip() or text[:200]
        except Exception:
            return text[:200]

    def _embed(self, text: str):
        import litellm
        try:
            r = litellm.embedding(model=EMBED_MODEL, input=[text[:2000]])
            return r.data[0]["embedding"]
        except Exception:
            return None

    @staticmethod
    def _render(msgs: list[dict]) -> str:
        out = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content")
            if m.get("tool_calls"):
                tc = m["tool_calls"][0]["function"]
                out.append(f"[assistant call] {tc['name']}({tc['arguments']})")
            elif content:
                out.append(f"[{role}] {content}")
        return "\n".join(out)

    def _finalize_blocks(self, rounds: list[dict]):
        """Summarize + embed rounds that have just become 'older' (cached once)."""
        older = rounds[:-self.recent_rounds] if len(rounds) > self.recent_rounds else []
        for b in older:
            if "summary" not in b:
                text = self._render(b["messages"])
                b["summary"] = self._summarize(text)
                b["emb"] = self._embed(b["summary"])

    def _build_context(self, system_msg, goal_msg, rounds, latest_text):
        recent = rounds[-self.recent_rounds:] if rounds else []
        older = rounds[:-self.recent_rounds] if len(rounds) > self.recent_rounds else []

        compressed = False
        selected = older
        if len(older) > self.n_blocks:
            compressed = True
            q = self._embed(latest_text)
            ranked = sorted(older, key=lambda b: -_cos(q, b.get("emb")))[: self.n_blocks]
            keep = set(id(b) for b in ranked)
            selected = [b for b in older if id(b) in keep]   # chronological order

        eff = [system_msg, goal_msg]
        summaries = [b.get("summary") for b in selected if b.get("summary")]
        if summaries:
            eff.append({
                "role": "user",
                "content": "## Earlier conversation (block-cleaned summary, most relevant):\n"
                           + "\n".join(f"- {s}" for s in summaries),
            })
        for b in recent:
            eff += b["messages"]
        return eff, compressed

    # ── Main loop (replaces ToolCallingAgent.solve, + mechanism 3) ────────────
    def run(self, task, env, task_index: int = -1) -> AgentRunResult:
        import litellm
        from tau_bench.agents.tool_calling_agent import message_to_action
        from tau_bench.types import RESPOND_ACTION_NAME

        gated = SaberGatedEnv(
            env=env, env_name=self.env_name,
            aux_model=self.aux_model, policy_text=get_policy_text(self.env_name),
        )
        tools_info = env.tools_info
        system_msg = {"role": "system", "content": env.wiki}

        reset_res = gated.reset(task_index)
        obs = getattr(reset_res, "observation", "")
        goal_msg = {"role": "user", "content": obs}

        rounds: list[dict] = []
        latest_text = obs
        total_cost = 0.0
        reward = 0.0
        n_compressed = 0

        for _ in range(MAX_NUM_STEPS):
            eff, compressed = self._build_context(system_msg, goal_msg, rounds, latest_text)
            if compressed:
                n_compressed += 1
            res = litellm.completion(
                messages=eff, model=self.model, custom_llm_provider=self.model_provider,
                tools=tools_info, temperature=0,
            )
            msg = res.choices[0].message.model_dump()
            total_cost += res._hidden_params.get("response_cost") or 0.0
            action = message_to_action(msg)
            env_response = gated.step(action)
            reward = env_response.reward

            if action.name != RESPOND_ACTION_NAME and msg.get("tool_calls"):
                msg["tool_calls"] = msg["tool_calls"][:1]
                result_msg = {
                    "role": "tool",
                    "tool_call_id": msg["tool_calls"][0]["id"],
                    "name": msg["tool_calls"][0]["function"]["name"],
                    "content": env_response.observation,
                }
            else:
                result_msg = {"role": "user", "content": env_response.observation}

            rounds.append({"messages": [msg, result_msg]})
            latest_text = env_response.observation
            self._finalize_blocks(rounds)
            if env_response.done:
                break

        logs = gated.step_logs
        return AgentRunResult(
            task_index=task_index,
            reward=float(reward),
            step_logs=logs,
            metadata={
                "method": "saber_block",
                "model": self.model,
                "aux_model": self.aux_model,
                "total_cost": total_cost,
                "num_verifications": gated.num_verifications,
                "num_executed_after_confirm": gated.num_executed_after_confirm,
                "num_user_rejected": gated.num_user_rejected,
                "num_aux_calls": gated.num_aux_calls,
                "num_blocks": len(rounds),
                "n_context_compressed_turns": n_compressed,
            },
        )
