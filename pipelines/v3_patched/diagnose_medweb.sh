#!/usr/bin/env bash
# diagnose_medweb.sh
# =============================================================================
# Crosses tau, entropy adaptation (adaptive_gamma) and query smoothing on
# MedWeb, four conditions per language. This run is the evidence behind
# docs/FIXES.md #9-#10, and condition 4 (tau=0.8, adaptive_gamma,
# smoothing=0.1) is the query-smoothing row of the paper's Table 3.
#
# LDA routing (tau=1.1) and RRF are not re-run here; use
# camera_ready_extras.py medweb for every baseline.
#
# Estimated time: ~3-5 min per condition on GPU (retrieval only, cache hit)
# =============================================================================

set -uo pipefail

mkdir -p experiment_logs1

MEDWEB_EN_PATH="../data/SOTA/ntcir13_MedWeb_TestCollection/medweb_rag_en_fixed.csv"
MEDWEB_JA_PATH="../data/SOTA/ntcir13_MedWeb_TestCollection/medweb_rag_ja_fixed.csv"
MEDWEB_ZH_PATH="../data/SOTA/ntcir13_MedWeb_TestCollection/medweb_rag_zh_fixed.csv"

FAILED_RUNS=()

run_experiment() {
    local label="$1"
    local log_file="$2"
    shift 2
    echo "  Running : $label"
    if "$@" > "$log_file" 2>&1; then
        mrr=$(grep "^MRR:" "$log_file" | head -1 | awk '{print $2}')
        echo "  ✓  MRR=$mrr  -> $log_file"
    else
        echo "  ✗  FAILED -> see $log_file"
        FAILED_RUNS+=("$label")
    fi
}

echo ""
echo "============================================================"
echo "  MedWeb diagnostic: tau x adaptive_gamma x query_smoothing"
echo "============================================================"

declare -A MEDWEB_PATHS=(
    [medweb_en]="$MEDWEB_EN_PATH"
    [medweb_ja]="$MEDWEB_JA_PATH"
    [medweb_zh]="$MEDWEB_ZH_PATH"
)

for dname in medweb_en medweb_ja medweb_zh; do
    dpath="${MEDWEB_PATHS[$dname]}"
    base=(--chunk_size 512 --num_topics 20 --gamma 0.7)

    if [ "$dname" = "medweb_ja" ]; then base+=(--chunk_size_ja 512); fi
    if [ "$dname" = "medweb_zh" ]; then base+=(--chunk_size_zh 512); fi

    echo ""
    echo "--- ${dname} ---"

    # CONDITION 1: tau=0.5, adaptive_gamma OFF, smoothing OFF
    run_experiment "${dname} | tau=0.5 | no adaptive_gamma | no smoothing" \
        "experiment_logs1/${dname}_diag_tau0.5_no_adaptive.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$dpath" \
        --tau 0.5 --query_smoothing 0.0 --no_adaptive_gamma \
        "${base[@]}"

    # CONDITION 2: tau=0.5, adaptive_gamma ON, smoothing OFF
    # Isolates: does smoothing help or hurt when combined with adaptive_gamma?
    run_experiment "${dname} | tau=0.5 | adaptive_gamma | no smoothing" \
        "experiment_logs1/${dname}_diag_tau0.5_adaptive_nosmooth.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$dpath" \
        --tau 0.5 --query_smoothing 0.0 --adaptive_gamma \
        "${base[@]}"

    # CONDITION 3: tau=0.8, no adaptive_gamma, no smoothing
    # Topic distributions for tau=0.8 are computed on first use (~5 min).
    run_experiment "${dname} | tau=0.8 | no adaptive_gamma | no smoothing" \
        "experiment_logs1/${dname}_diag_tau0.8_no_adaptive.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$dpath" \
        --tau 0.8 --query_smoothing 0.0 --no_adaptive_gamma \
        "${base[@]}"

    # CONDITION 4: tau=0.8, adaptive_gamma ON, smoothing=0.1
    # Reuses the topic distributions of condition 3.
    run_experiment "${dname} | tau=0.8 | adaptive_gamma | smoothing=0.1" \
        "experiment_logs1/${dname}_diag_tau0.8_adaptive_smooth.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$dpath" \
        --tau 0.8 --query_smoothing 0.1 --adaptive_gamma \
        "${base[@]}"

done

# ---------------------------------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  DIAGNOSTIC RESULTS"
echo "============================================================"
printf "%-12s  %-8s  %-8s  %-8s  %-8s\n" \
    "Dataset" "τ=0.5" "τ=0.5+ag" "τ=0.8" "τ=0.8+ag"
echo "-----------------------------------------------------------"
printf "%-12s  %-8s  %-8s  %-8s  %-8s  (reference)\n" \
    "" "no fixes" "no smooth" "no fixes" "+smooth"
echo "-----------------------------------------------------------"

for dname in medweb_en medweb_ja medweb_zh; do
    c1=$(grep "^MRR:" "experiment_logs1/${dname}_diag_tau0.5_no_adaptive.log"      2>/dev/null | head -1 | awk '{print $2}')
    c2=$(grep "^MRR:" "experiment_logs1/${dname}_diag_tau0.5_adaptive_nosmooth.log" 2>/dev/null | head -1 | awk '{print $2}')
    c3=$(grep "^MRR:" "experiment_logs1/${dname}_diag_tau0.8_no_adaptive.log"       2>/dev/null | head -1 | awk '{print $2}')
    c4=$(grep "^MRR:" "experiment_logs1/${dname}_diag_tau0.8_adaptive_smooth.log"   2>/dev/null | head -1 | awk '{print $2}')
    printf "%-12s  %-8s  %-8s  %-8s  %-8s\n" \
        "$dname" "${c1:--}" "${c2:--}" "${c3:--}" "${c4:--}"
done

echo "-----------------------------------------------------------"
echo "============================================================"
echo "============================================================"

if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "  FAILED RUNS:"
    for run in "${FAILED_RUNS[@]}"; do echo "    - $run"; done
fi
