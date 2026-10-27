#!/usr/bin/env bash
#
# BehaviorICL synthetic demo — end-to-end pipeline run.
#
#   ./run_demo.sh                 all nine traits
#   ./run_demo.sh cooperation     one trait
#   ./run_demo.sh trust altruism  a couple of traits
#
# Set BEHAVIORICL_MOCK=1 to run with no API key and no spend. The numbers that
# come out in that mode are a plumbing test, not a finding — see README.md and
# the warning at the top of 2_Inference/mock_backend.py.
#
# Otherwise set OPENAI_API_KEY and the pipeline will call BEHAVIORICL_MODEL
# (default gpt-4o; see 2_Inference/client.py).

set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(pwd)
PYTHON=${PYTHON:-python3}

ALL_TRAITS=(trust competitiveness cooperation coordination depth_of_reasoning
            tolerance_for_inequality risk_tolerance altruism morality)

if [ "$#" -gt 0 ]; then
    TRAITS=("$@")
else
    TRAITS=("${ALL_TRAITS[@]}")
fi

if [ "${BEHAVIORICL_MOCK:-}" = "" ] && [ "${OPENAI_API_KEY:-}" = "" ]; then
    echo "Neither BEHAVIORICL_MOCK nor OPENAI_API_KEY is set."
    echo "For a free offline run:  export BEHAVIORICL_MOCK=1"
    exit 1
fi

if [ "${BEHAVIORICL_MOCK:-}" != "" ]; then
    echo "### MOCK MODE — offline heuristic, not a model. Results are a smoke test only."
fi

banner () { printf '\n\033[1m=== %s\033[0m\n' "$1"; }

banner "0. Generate synthetic data"
"$PYTHON" 0_Preprocess/make_synthetic_data.py

banner "0. Validity matrix (hold-out eligibility per app)"
cd "$ROOT/0_Preprocess" && "$PYTHON" validity_matrix.py

for trait in "${TRAITS[@]}"; do
    banner "1. Build prompts — $trait"
    cd "$ROOT/1_Prompts" && "$PYTHON" build_prompts.py "$trait" | head -n 6

    banner "2. Inference (ICL / zero-shot / other-student ICL) — $trait"
    cd "$ROOT/2_Inference" && "$PYTHON" run_trait.py "$trait"

    banner "3. Evaluate — $trait"
    cd "$ROOT/3_Eval" && "$PYTHON" eval_trait.py "$trait"
done

if [ "${#TRAITS[@]}" -eq "${#ALL_TRAITS[@]}" ]; then
    banner "3. Cross-trait report and baselines"
    cd "$ROOT/3_Eval"
    "$PYTHON" cross_trait/build_report.py
    "$PYTHON" pca_baseline.py
    "$PYTHON" ml_benchmarks.py
fi

banner "Done"
echo "Prompts   : 1_Prompts/<trait>/prompts.jsonl"
echo "Raw output: 2_Inference/<trait>/*.jsonl"
echo "Metrics   : 3_Eval/<trait>/summary.txt, validation_stats.txt"
