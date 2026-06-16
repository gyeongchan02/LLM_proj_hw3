"""
Divergence labeler — P2 deliverable (README §1.3).

A task "diverges" when, somewhere in a run, a *mutating* action first departs
from the correct trajectory. The first such mutating step is the divergence
point. Recovery Rate (metrics.compute_recovery_rate) is measured over the set
of task_index values that diverged: of those, how many still ended with
reward > 0.

Three modes (pick by data available):

  pairwise   compare a SUCCESS run log and a FAILURE run log for the same task;
             the first mutating step where their action sequences differ is the
             divergence point. This is the canonical README §1.3 method.

  vs-gold    compare ONE run log against the tau-bench gold action sequence;
             the first mutating step that deviates from gold is the divergence.
             Useful when only one run is available (gold = perfect success ref).

  bootstrap  no run logs needed: derive divergence tasks from the perturbation
             labels (every task carrying a non-approve judgment point has a
             known mistake opportunity). Lets P3 compute recovery_rate before
             any experiment has run; replace with pairwise/vs-gold once real
             logs exist.

Output: data/labels/divergences.jsonl, one object per diverged task:
    {"task_index": 7, "divergence_step": 3, "tool": "cancel_pending_order"}

Usage:
    # from real run logs vs gold
    python -m src.eval.label_divergence vs-gold \
        --run data/results/vanilla_seed42.jsonl \
        --env retail --split test \
        --out ../data/labels/divergences.jsonl

    # success vs failure run logs
    python -m src.eval.label_divergence pairwise \
        --success data/results/ours_seed42.jsonl \
        --failure data/results/vanilla_seed42.jsonl \
        --out ../data/labels/divergences.jsonl

    # bootstrap from perturbation labels (no runs yet)
    python -m src.eval.label_divergence bootstrap \
        --perturbations ../data/labels/perturbations.jsonl \
        --out ../data/labels/divergences.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional


# ── Action canonicalization ──────────────────────────────────────────────────

def _action_key(tool: str, args: dict) -> tuple:
    """Order-insensitive identity of a mutating action for comparison."""
    items = tuple(sorted((str(k), json.dumps(v, sort_keys=True, default=str))
                          for k, v in (args or {}).items()))
    return (tool, items)


def _mutating_steps_from_log(result: dict) -> list[tuple[int, str, dict]]:
    """Extract (step, tool, args) for mutating steps from a result's step_logs."""
    out = []
    for log in result.get("step_logs", []):
        if log.get("is_mutating"):
            out.append((log["step"], log["tool"], log.get("args", {})))
    return out


def _mutating_steps_from_gold(task) -> list[tuple[int, str, dict]]:
    """(step, tool, args) for mutating gold actions, step = 1-based position."""
    from src.data.action_taxonomy import is_mutating
    out = []
    for step, a in enumerate(task.actions, start=1):
        if is_mutating(a.name):
            out.append((step, a.name, dict(a.kwargs or {})))
    return out


def _first_divergence(
    run: list[tuple[int, str, dict]],
    ref: list[tuple[int, str, dict]],
) -> Optional[tuple[int, str]]:
    """First mutating step in `run` whose action differs from `ref` at the same
    ordinal position. Returns (run_step, tool) or None if run matches a prefix
    of ref (no divergence among the steps actually taken)."""
    for i, (step, tool, args) in enumerate(run):
        if i >= len(ref):
            return (step, tool)                    # run did an extra mutation
        if _action_key(tool, args) != _action_key(ref[i][1], ref[i][2]):
            return (step, tool)
    return None


# ── Modes ────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def mode_vs_gold(run_path: str, env: str, split: str) -> list[dict]:
    from src.data.perturb import load_tasks
    tasks = load_tasks(env, split)
    runs = _load_jsonl(run_path)
    labels = []
    for r in runs:
        idx = r["task_index"]
        if idx >= len(tasks):
            continue
        run_steps = _mutating_steps_from_log(r)
        gold_steps = _mutating_steps_from_gold(tasks[idx])
        div = _first_divergence(run_steps, gold_steps)
        if div is not None:
            labels.append({"task_index": idx,
                           "divergence_step": div[0], "tool": div[1]})
    return labels


def mode_pairwise(success_path: str, failure_path: str) -> list[dict]:
    succ = {r["task_index"]: r for r in _load_jsonl(success_path)}
    fail = {r["task_index"]: r for r in _load_jsonl(failure_path)}
    labels = []
    for idx in sorted(set(succ) & set(fail)):
        s_steps = _mutating_steps_from_log(succ[idx])
        f_steps = _mutating_steps_from_log(fail[idx])
        div = _first_divergence(f_steps, s_steps)   # where failure left success
        if div is not None:
            labels.append({"task_index": idx,
                           "divergence_step": div[0], "tool": div[1]})
    return labels


def mode_bootstrap(perturbations_path: str) -> list[dict]:
    rows = _load_jsonl(perturbations_path)
    # Earliest non-approve judgment point per task.
    best: dict[int, dict] = {}
    for r in rows:
        if r.get("gold_decision") == "approve":
            continue
        idx = r["task_index"]
        step = r.get("step", 1)
        if idx not in best or step < best[idx]["divergence_step"]:
            best[idx] = {"task_index": idx, "divergence_step": step,
                         "tool": r["tool"]}
    return [best[i] for i in sorted(best)]


# ── CLI ──────────────────────────────────────────────────────────────────────

def _write(labels: list[dict], out: str):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        for row in labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(labels)} divergence labels → {out}")


def _cli():
    ap = argparse.ArgumentParser(description="Label divergence points")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_g = sub.add_parser("vs-gold")
    p_g.add_argument("--run", required=True)
    p_g.add_argument("--env", default="retail")
    p_g.add_argument("--split", default="test")
    p_g.add_argument("--out", required=True)

    p_p = sub.add_parser("pairwise")
    p_p.add_argument("--success", required=True)
    p_p.add_argument("--failure", required=True)
    p_p.add_argument("--out", required=True)

    p_b = sub.add_parser("bootstrap")
    p_b.add_argument("--perturbations", required=True)
    p_b.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.mode == "vs-gold":
        labels = mode_vs_gold(args.run, args.env, args.split)
    elif args.mode == "pairwise":
        labels = mode_pairwise(args.success, args.failure)
    elif args.mode == "bootstrap":
        labels = mode_bootstrap(args.perturbations)
    else:
        raise ValueError(args.mode)
    _write(labels, args.out)


if __name__ == "__main__":
    _cli()
