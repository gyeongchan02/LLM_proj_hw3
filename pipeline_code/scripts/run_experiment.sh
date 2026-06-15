#!/usr/bin/env bash
# run_experiment.sh — single entry point for all experiments
#
# Usage:
#   ./scripts/run_experiment.sh [OPTIONS]
#
# Options:
#   --config PATH          YAML config file          [default: configs/experiment.yaml]
#   --method NAME [...]    Methods to run            [default: all in config]
#   --start-index N        First task index          [default: from config]
#   --end-index N          Last task index (excl.)   [default: from config]
#   --output-dir PATH      Where to write JSONL      [default: data/results]
#   --compute-metrics      Compute metrics after run [flag]
#   --gold-labels PATH     P2's perturbation JSONL   [for metrics]
#   --divergence-file PATH P2's divergence JSONL     [for recovery rate]
#
# Examples:
#   # Quick smoke test (5 tasks, vanilla only):
#   ./scripts/run_experiment.sh --method vanilla --end-index 5
#
#   # Full comparison run:
#   ./scripts/run_experiment.sh --config configs/experiment.yaml
#
#   # Single method with metrics:
#   ./scripts/run_experiment.sh --method ours --end-index 20 \
#       --compute-metrics \
#       --gold-labels data/labels/perturbations.jsonl \
#       --divergence-file data/labels/divergences.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── defaults ────────────────────────────────────────────────────────────────
CONFIG="configs/experiment.yaml"
METHODS=()
START_INDEX=""
END_INDEX=""
OUTPUT_DIR=""
COMPUTE_METRICS=false
GOLD_LABELS=""
DIVERGENCE_FILE=""
METRICS_OUTPUT="data/results/metrics.csv"

# ── parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)       CONFIG="$2";         shift 2 ;;
    --method)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        METHODS+=("$1"); shift
      done
      ;;
    --start-index)  START_INDEX="$2";   shift 2 ;;
    --end-index)    END_INDEX="$2";     shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2";    shift 2 ;;
    --compute-metrics) COMPUTE_METRICS=true; shift ;;
    --gold-labels)  GOLD_LABELS="$2";   shift 2 ;;
    --divergence-file) DIVERGENCE_FILE="$2"; shift 2 ;;
    --metrics-output) METRICS_OUTPUT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ── env check ────────────────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env | xargs)
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY not set. Add it to .env or export it." >&2
  exit 1
fi

# ── build runner args ────────────────────────────────────────────────────────
RUNNER_ARGS=("--config" "$CONFIG")
if [[ ${#METHODS[@]} -gt 0 ]]; then
  RUNNER_ARGS+=("--method" "${METHODS[@]}")
fi
if [[ -n "$START_INDEX" ]]; then
  RUNNER_ARGS+=("--start-index" "$START_INDEX")
fi
if [[ -n "$END_INDEX" ]]; then
  RUNNER_ARGS+=("--end-index" "$END_INDEX")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  RUNNER_ARGS+=("--output-dir" "$OUTPUT_DIR")
fi

# ── run experiments ───────────────────────────────────────────────────────────
echo "=== Starting experiment run ==="
echo "Config: $CONFIG"
python -m src.harness.runner "${RUNNER_ARGS[@]}"

# ── compute metrics (optional) ────────────────────────────────────────────────
if $COMPUTE_METRICS; then
  ACTUAL_OUTPUT="${OUTPUT_DIR:-data/results}"
  METRICS_ARGS=("--results-dir" "$ACTUAL_OUTPUT" "--output" "$METRICS_OUTPUT")
  if [[ -n "$GOLD_LABELS" ]];     then METRICS_ARGS+=("--gold-labels" "$GOLD_LABELS"); fi
  if [[ -n "$DIVERGENCE_FILE" ]]; then METRICS_ARGS+=("--divergence-file" "$DIVERGENCE_FILE"); fi

  echo ""
  echo "=== Computing metrics ==="
  python -m src.eval.metrics "${METRICS_ARGS[@]}"
fi

echo ""
echo "=== Done ==="
