#!/usr/bin/env bash
# run_gpt4o_cumulative.sh - Run cumulative iterations and evaluate
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/third_party:${PYTHONPATH:-}"
test -f "$ROOT_DIR/third_party/PaTAT_piped/__init__.py" || touch "$ROOT_DIR/third_party/PaTAT_piped/__init__.py"
[ -e "$ROOT_DIR/third_party/synthesizer" ] || ln -sfn "$ROOT_DIR/third_party/PaTAT_piped" "$ROOT_DIR/third_party/synthesizer"

# Check for Azure OpenAI configuration
if [ ! -f "config.ini" ]; then
    echo "ERROR: config.ini not found"
    exit 1
fi

# Fixed seed as requested
SEED=42
export SEED

echo "================================================================================"
echo "CUMULATIVE ITERATION PIPELINE (Azure OpenAI)"
echo "================================================================================"
echo "Seed: ${SEED}"
echo "This will run Steps 01-04 for iterations 0-10, then Step 05 for evaluation"
echo "================================================================================"
echo ""

# Helper function to run python
run_py() {
    local script="$1"
    python "$script"
}

# Read dataset from config
DATA_FILE=$(python - <<'PY'
import configparser
c=configparser.ConfigParser(inline_comment_prefixes=('#',';'))
c.read('config.ini')
print(c.get('settings','data_file').strip())
PY
)

echo "Dataset file: ${DATA_FILE}"
echo ""

# Run iterations 0-10
for ITER in {0..10}; do
    echo "================================================================================"
    echo "ITERATION ${ITER}"
    echo "================================================================================"
    export ITERATION=${ITER}

    echo "Step 01: Data formatting for iteration ${ITER}"
    if ! run_py 01_data_formatting.py; then
        echo "ERROR: Step 01 failed for iteration ${ITER}"
        exit 1
    fi

    echo ""
    echo "Step 02: Counterfactual generation for iteration ${ITER}"
    if ! run_py 02_counterfactual_over_generation.py; then
        echo "ERROR: Step 02 failed for iteration ${ITER}"
        exit 1
    fi

    echo ""
    echo "Step 03: Counterfactual filtering for iteration ${ITER}"
    if ! run_py 03_counterfactual_filtering.py; then
        echo "ERROR: Step 03 failed for iteration ${ITER}"
        exit 1
    fi

    echo ""
    echo "Step 04: Fine-tuning for iteration ${ITER}"
    if ! run_py 04_fine_tuning.py; then
        echo "WARNING: Step 04 failed for iteration ${ITER} (continuing anyway)"
    fi

    echo ""
    echo "✓ Iteration ${ITER} complete"
    echo ""
done

echo "================================================================================"
echo "ALL ITERATIONS COMPLETE - RUNNING CUMULATIVE EVALUATION"
echo "================================================================================"
echo ""

# Run evaluation across all cumulative iterations
echo "Step 05: Cumulative evaluation across all iterations"
if ! run_py 05_AL_testing.py; then
    echo "ERROR: Step 05 (evaluation) failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: output_data/results/"
echo "  - Detailed results: [${SEED}]cumulative_results_*.csv"
echo "  - Summary table: [${SEED}]summary_*.csv"
echo "  - Formatted summary: [${SEED}]summary_*.txt"
echo ""
echo "================================================================================"