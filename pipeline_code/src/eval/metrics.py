"""
Metrics for P3.

Inputs: JSONL result files produced by runner.py
Outputs: dict / CSV / printed table

Usage:
    from src.eval.metrics import compute_all_metrics, load_results
    results = load_results("data/results/ours_seed42.jsonl")
    m = compute_all_metrics(results)

OR via CLI:
    python -m src.eval.metrics --results-dir data/results --output metrics.csv

P3 responsibilities:
  - Run this after all experiment JSONL files exist.
  - Compute divergence_task_indices using label_divergence.py (P2 provides).
  - Feed divergence_task_indices to compute_recovery_rate().
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> list[dict]:
    """Load one JSONL result file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def load_all_results(results_dir: str) -> dict[str, list[dict]]:
    """
    Load all result files in results_dir.
    Returns {method_name: [result_dict, ...]}.
    Files must be named {method}_seed{N}.jsonl.
    """
    all_results: dict[str, list[dict]] = {}
    for p in sorted(Path(results_dir).glob("*.jsonl")):
        method = p.stem.split("_seed")[0]
        records = load_results(str(p))
        all_results.setdefault(method, []).extend(records)
    return all_results


# ---------------------------------------------------------------------------
# Task-level metrics
# ---------------------------------------------------------------------------

def pass_at_1(results: list[dict]) -> float:
    """Fraction of tasks with reward > 0 on the first (or only) attempt."""
    if not results:
        return 0.0
    return sum(1 for r in results if r["reward"] > 0) / len(results)


def pass_k(results_by_task: dict[int, list[dict]]) -> float:
    """
    pass^k: fraction of tasks where ALL k attempts succeed.
    results_by_task: {task_index: [result_attempt_1, result_attempt_2, ...]}
    """
    if not results_by_task:
        return 0.0
    all_success = sum(
        1 for attempts in results_by_task.values()
        if all(r["reward"] > 0 for r in attempts)
    )
    return all_success / len(results_by_task)


def compute_recovery_rate(
    results: list[dict],
    divergence_task_indices: set[int],
) -> float:
    """
    Recovery Rate: among tasks that HAD a divergence point (mutating mistake),
    what fraction ended with reward > 0?

    divergence_task_indices: set of task_index values where divergence was found.
    Provided by P2's label_divergence.py applied to the run logs.
    """
    if not divergence_task_indices:
        return float("nan")
    diverged = [r for r in results if r["task_index"] in divergence_task_indices]
    if not diverged:
        return float("nan")
    return sum(1 for r in diverged if r["reward"] > 0) / len(diverged)


# ---------------------------------------------------------------------------
# Critic-accuracy metrics (require P2's gold labels)
# ---------------------------------------------------------------------------

def compute_critic_metrics(
    results: list[dict],
    gold_labels: list[dict],
) -> dict[str, float]:
    """
    Compare critic decisions in step_logs against P2's gold labels.

    gold_labels: list of dicts from P2's perturbation JSONL:
      {"task_index": 12, "tool": "cancel_pending_order",
       "gold_decision": "block", "reversible": true, ...}

    Returns precision, recall, false_block_rate, 4_way_accuracy, reversibility_accuracy.
    """
    # Build lookup: (task_index, tool) → gold entry
    gold_map: dict[tuple[int, str], dict] = {}
    for g in gold_labels:
        key = (g["task_index"], g["tool"])
        gold_map[key] = g

    tp = fp = fn = tn = 0
    correct_verdict = total_verdict = 0
    correct_rev = total_rev = 0

    for r in results:
        task_idx = r["task_index"]
        for log in r.get("step_logs", []):
            if not log.get("is_mutating"):
                continue
            key = (task_idx, log["tool"])
            if key not in gold_map:
                continue
            gold = gold_map[key]
            pred_verdict = log.get("decision") or "approve"
            gold_verdict = gold["gold_decision"]

            # 4-way accuracy
            total_verdict += 1
            if pred_verdict == gold_verdict:
                correct_verdict += 1

            # Binary block detection (block vs non-block)
            pred_block = pred_verdict == "block"
            gold_block = gold_verdict == "block"
            if pred_block and gold_block:
                tp += 1
            elif pred_block and not gold_block:
                fp += 1
            elif not pred_block and gold_block:
                fn += 1
            else:
                tn += 1

            # Reversibility
            if "reversible" in gold and log.get("reversible") is not None:
                total_rev += 1
                if log["reversible"] == gold["reversible"]:
                    correct_rev += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    false_block_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    accuracy_4way = correct_verdict / total_verdict if total_verdict > 0 else float("nan")
    rev_accuracy = correct_rev / total_rev if total_rev > 0 else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "false_block_rate": false_block_rate,
        "4way_accuracy": accuracy_4way,
        "reversibility_accuracy": rev_accuracy,
    }


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def compute_latency_stats(results: list[dict]) -> dict[str, float]:
    times = [r["metadata"].get("elapsed_s", 0) for r in results if "metadata" in r]
    if not times:
        return {"mean_s": float("nan"), "median_s": float("nan")}
    times.sort()
    n = len(times)
    return {
        "mean_s": sum(times) / n,
        "median_s": times[n // 2],
        "p95_s": times[int(n * 0.95)],
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results: list[dict],
    divergence_task_indices: set[int] | None = None,
    gold_labels: list[dict] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics["n"] = len(results)
    metrics["pass@1"] = pass_at_1(results)

    if divergence_task_indices is not None:
        metrics["recovery_rate"] = compute_recovery_rate(results, divergence_task_indices)

    if gold_labels is not None:
        metrics.update(compute_critic_metrics(results, gold_labels))

    metrics.update(compute_latency_stats(results))

    # Gating summary (from step logs)
    total_mutating = total_blocked = total_revised = total_ask = 0
    for r in results:
        for log in r.get("step_logs", []):
            if log.get("is_mutating"):
                total_mutating += 1
                d = log.get("decision", "approve")
                if d == "block":
                    total_blocked += 1
                elif d == "revise":
                    total_revised += 1
                elif d == "ask_user":
                    total_ask += 1
    metrics["total_mutating_steps"] = total_mutating
    metrics["num_blocked"] = total_blocked
    metrics["num_revised"] = total_revised
    metrics["num_ask_user"] = total_ask

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Compute experiment metrics")
    parser.add_argument("--results-dir", required=True, help="Dir with *.jsonl result files")
    parser.add_argument("--gold-labels", default=None, help="P2's perturbation JSONL (for critic metrics)")
    parser.add_argument("--divergence-file", default=None, help="P2's divergence label JSONL")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    gold_labels = None
    if args.gold_labels:
        with open(args.gold_labels) as f:
            gold_labels = [json.loads(l) for l in f if l.strip()]

    divergence_task_indices = None
    if args.divergence_file:
        with open(args.divergence_file) as f:
            divergence_task_indices = {
                json.loads(l)["task_index"] for l in f if l.strip()
            }

    all_results = load_all_results(args.results_dir)

    rows = []
    for method, results in sorted(all_results.items()):
        m = compute_all_metrics(results, divergence_task_indices, gold_labels)
        m["method"] = method
        rows.append(m)
        print(f"\n[{method}]")
        for k, v in m.items():
            if k != "method":
                print(f"  {k}: {_fmt(v)}")

    if args.output and rows:
        keys = list(rows[0].keys())
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved → {args.output}")


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "N/A"
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    _cli()
