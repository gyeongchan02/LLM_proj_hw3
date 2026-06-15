"""
Experiment runner.

Usage (programmatic):
    from src.harness.runner import run_experiment
    results = run_experiment(config)

Usage (CLI via run_experiment.sh):
    python -m src.harness.runner --config configs/experiment.yaml --method ours

Output: one JSONL file per (method, seed) under data/results/.
Each line is the serialised AgentRunResult for one task.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.agents import get_agent
from src.critic.schemas import AgentRunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_experiment(config: dict) -> dict[str, list[AgentRunResult]]:
    """
    Run all methods defined in config["methods"] for all seeds.
    Returns {method_name: [AgentRunResult, ...]} across all seeds.
    """
    env_name: str = config.get("env_name", "retail")
    task_split: str = config.get("task_split", "test")
    start_index: int = config.get("start_index", 0)
    end_index: int | None = config.get("end_index", None)
    seeds: list[int] = config.get("seeds", [42])
    output_dir: str = config.get("output_dir", "data/results")
    methods: list[dict] = config.get("methods", [])

    os.makedirs(output_dir, exist_ok=True)

    all_results: dict[str, list[AgentRunResult]] = {}

    for method_cfg in methods:
        method_name: str = method_cfg["name"]
        agent_kwargs: dict = _build_agent_kwargs(method_cfg, config)

        all_results[method_name] = []

        for seed in seeds:
            _set_seed(seed)
            env = _build_env(env_name, task_split, config, seed)
            tasks, indices = _get_tasks(env, start_index, end_index)

            agent = get_agent(method=method_name, **agent_kwargs)

            run_results: list[AgentRunResult] = []
            for task, idx in zip(tasks, indices):
                logger.info(f"[{method_name}] task {idx}")
                t0 = time.time()
                try:
                    result = agent.run(task=task, env=env, task_index=idx)
                except Exception as e:
                    logger.error(f"Task {idx} failed: {e}")
                    result = AgentRunResult(
                        task_index=idx,
                        reward=0.0,
                        metadata={"method": method_name, "error": str(e)},
                    )
                result.metadata["elapsed_s"] = round(time.time() - t0, 2)
                result.metadata["seed"] = seed
                run_results.append(result)
                all_results[method_name].append(result)

            _save_results(run_results, method_name, seed, output_dir)

    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_env(env_name: str, task_split: str, config: dict, seed: int):
    """
    Build tau-bench environment.

    Tau-bench compatibility note:
    The exact get_env() signature may differ by version. If you see errors,
    check: from tau_bench.envs import get_env; help(get_env)
    """
    try:
        from tau_bench.envs import get_env
        user_cfg = config.get("user", {})
        return get_env(
            env_name=env_name,
            user_strategy=user_cfg.get("strategy", "react"),
            user_model=user_cfg.get("model", "gpt-4o"),
            user_model_provider=user_cfg.get("provider", "openai"),
            task_split=task_split,
        )
    except ImportError:
        raise RuntimeError(
            "tau-bench not installed.\n"
            "Run: pip install -e external/tau-bench\n"
            "Or:  git submodule add https://github.com/sierra-research/tau-bench external/tau-bench"
        )


def _get_tasks(env, start: int, end: int | None) -> tuple[list, list[int]]:
    """
    Retrieve tasks from tau-bench env.

    Tau-bench compatibility note:
    Different tau-bench versions expose tasks differently. Try each approach.
    """
    # Approach 1: env.get_all_tasks() or env.tasks
    if hasattr(env, "get_all_tasks"):
        all_tasks = env.get_all_tasks()
    elif hasattr(env, "tasks"):
        all_tasks = env.tasks
    else:
        raise AttributeError(
            "Cannot find task list in env. "
            "Check tau-bench version: env should have .tasks or .get_all_tasks()"
        )

    actual_end = end if end is not None else len(all_tasks)
    tasks = all_tasks[start:actual_end]
    indices = list(range(start, start + len(tasks)))
    return tasks, indices


def _build_agent_kwargs(method_cfg: dict, global_cfg: dict) -> dict:
    """Merge method-specific config with global defaults."""
    main_model = method_cfg.get("model", global_cfg.get("model", "gpt-4o"))
    main_provider = method_cfg.get("model_provider", global_cfg.get("model_provider", "openai"))
    env_name = global_cfg.get("env_name", "retail")

    base = {
        "model": main_model,
        "model_provider": main_provider,
        "env_name": env_name,
    }

    name = method_cfg["name"]

    if name == "ours":
        base["critic_model"] = method_cfg.get(
            "critic_model", global_cfg.get("critic_model", "claude-3-5-sonnet-latest")
        )
        base["condition_ablation"] = method_cfg.get("condition_ablation")
        base["enable_rollback"] = method_cfg.get("enable_rollback", False)

    elif name == "oracle":
        base["label_file"] = method_cfg.get(
            "label_file", global_cfg.get("oracle_label_file", "data/labels/perturbations.jsonl")
        )

    elif name == "reflexion":
        base["max_reflections"] = method_cfg.get("max_reflections", 3)

    return base


def _save_results(results: list[AgentRunResult], method: str, seed: int, output_dir: str):
    fname = Path(output_dir) / f"{method}_seed{seed}.jsonl"
    with open(fname, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    logger.info(f"Saved {len(results)} results → {fname}")


def _set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(description="Run critic-gating experiment")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--method", nargs="*",
        help="Override which methods to run (space-separated). Default: all from config.",
    )
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.method:
        config["methods"] = [m for m in config["methods"] if m["name"] in args.method]
    if args.start_index is not None:
        config["start_index"] = args.start_index
    if args.end_index is not None:
        config["end_index"] = args.end_index
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run_experiment(config)

    # Quick summary
    print("\n=== Run complete ===")
    for method, runs in results.items():
        avg = sum(r.reward for r in runs) / max(len(runs), 1)
        print(f"  {method}: n={len(runs)}, avg_reward={avg:.3f}")


if __name__ == "__main__":
    _cli()
