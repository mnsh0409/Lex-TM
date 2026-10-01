#!/usr/bin/env bash
# run_all.sh
# =============================================================================
# Master script — runs the earlier-pipeline experiments of the AACL paper.
#
# PREREQUISITES
# -------------
# You must have existing caches in experiment_logs1/ (K=50 for non-MedWeb,
# K=20 for MedWeb). This script will:
#   1. Copy K=50 embeddings to experiment_logs/ (for validation pipeline)
#   2. Run MedWeb (EN/JA/ZH) via patched pipeline — Lex-TM + RRF
#   3. Run all other datasets via patched pipeline — Lex-TM + RRF
#
# CACHE BEHAVIOUR PER DATASET
# ---------------------------
#   MedWeb    (K=20): embeddings load from cache; topicdists recompute if
#                     tau changes. ~5 min per language on GPU.
#   Others    (K=50): both embeddings AND topicdists load from cache.
#                     Evaluation only — very fast (~1-2 min per dataset).
#
# WHICH SCRIPT USES WHICH CACHE DIR
# ----------------------------------
#   main_pipeline1_patched.py  →  experiment_logs1/   (original dir, no copy)
#   main_pipeline_valid.py     →  experiment_logs/     (needs copy_cache.sh)
#
# USAGE
# -----
#   bash run_all.sh                    # full run
#   bash run_all.sh --skip-copy        # skip cache copy (already done)
#   bash run_all.sh --medweb-only      # MedWeb only
#   bash run_all.sh --skip-medweb      # everything except MedWeb
# =============================================================================

set -uo pipefail

mkdir -p experiment_logs1 experiment_logs

# ---------------------------------------------------------------------------
# ARGUMENT PARSING
# ---------------------------------------------------------------------------
SKIP_COPY=false
MEDWEB_ONLY=false
SKIP_MEDWEB=false

for arg in "$@"; do
    case $arg in
        --skip-copy)    SKIP_COPY=true    ;;
        --medweb-only)  MEDWEB_ONLY=true  ;;
        --skip-medweb)  SKIP_MEDWEB=true  ;;
    esac
done

FAILED_RUNS=()

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
MEDWEB_EN_PATH="../data/SOTA/ntcir13_MedWeb_TestCollection/medweb_rag_en_fixed.csv"
MEDWEB_JA_PATH="../data/SOTA/ntcir13_MedWeb_TestCollection/medweb_rag_ja_fixed.csv"
MEDWEB_ZH_PATH="../data/SOTA/ntcir13_MedWeb_TestCollection/medweb_rag_zh_fixed.csv"

EN_AVIATION_PATH="../data/Top10_airlines_reviews(2016-2023Jul).txt"
ZH_AVIATION_PATH="../data/translated_reviews_multi_language.csv"

MIRACL_ZH_CORPUS="../data/SOTA/miracl_zh/corpus.jsonl"
MIRACL_ZH_QUERIES="../data/SOTA/miracl_zh/queries.jsonl"
MIRACL_ZH_QRELS="../data/SOTA/miracl_zh/qrels/dev.tsv"

MR_TYDI_JA_CORPUS="../data/SOTA/mr_tydi_ja/corpus.jsonl"
MR_TYDI_JA_QUERIES="../data/SOTA/mr_tydi_ja/queries.jsonl"
MR_TYDI_JA_QRELS="../data/SOTA/mr_tydi_ja/qrels/test.tsv"

MR_TYDI_TH_CORPUS="../data/SOTA/mr_tydi_th/corpus.jsonl"
MR_TYDI_TH_QUERIES="../data/SOTA/mr_tydi_th/queries.jsonl"
MR_TYDI_TH_QRELS="../data/SOTA/mr_tydi_th/qrels/test.tsv"

THUCNEWS_PATH="../data/SOTA/thucnews1.csv"
SOGOU_PATH="../data/SOTA/sogou_news1.csv"

NFCORPUS_CORPUS="../data/beir/nfcorpus/corpus.jsonl"
NFCORPUS_QUERIES="../data/beir/nfcorpus/queries.jsonl"
NFCORPUS_QRELS="../data/beir/nfcorpus/qrels/test.tsv"

TREC_COVID_CORPUS="../data/beir/trec_covid/corpus.jsonl"
TREC_COVID_QUERIES="../data/beir/trec_covid/queries.jsonl"
TREC_COVID_QRELS="../data/beir/trec_covid/qrels/test.tsv"

# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
run_experiment() {
    local label="$1"
    local log_file="$2"
    shift 2
    echo "  Running : $label"
    if "$@" > "$log_file" 2>&1; then
        echo "  ✓  Done  -> $log_file"
        grep -E "^MRR:" "$log_file" | sed 's/^/         /'
    else
        echo "  ✗  FAILED (exit $?) -> see $log_file"
        FAILED_RUNS+=("$label")
    fi
}

# ---------------------------------------------------------------------------
# STEP 1: COPY EMBEDDINGS CACHE
# experiment_logs1/ → experiment_logs/ (for validation pipeline)
# Only copies _embeddings.npy — _topicdists.npy are K-dependent
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STEP 1: Copy embeddings cache"
echo "============================================================"

if [ "$SKIP_COPY" = true ]; then
    echo "  Skipped (--skip-copy)"
else
    bash copy_cache.sh
fi

if [ "$MEDWEB_ONLY" = true ]; then
    echo ""
    echo "  --medweb-only: skipping to Section 1"
fi

# ---------------------------------------------------------------------------
# STEP 2: MEDWEB — PRIMARY BENCHMARK
# Uses patched pipeline (experiment_logs1/ cache, fixes 3 & 4 active)
# MedWeb uses K=20 — may recompute topicdists if not cached at this tau
# --run_bm25 produces RRF numbers alongside Lex-TM in the same run
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STEP 2: MedWeb (EN / JA / ZH) — Primary Benchmark"
echo "============================================================"

if [ "$SKIP_MEDWEB" = false ]; then
    declare -A MEDWEB_PATHS=(
        [medweb_en]="$MEDWEB_EN_PATH"
        [medweb_ja]="$MEDWEB_JA_PATH"
        [medweb_zh]="$MEDWEB_ZH_PATH"
    )

    for dname in medweb_en medweb_ja medweb_zh; do
        dpath="${MEDWEB_PATHS[$dname]}"
        # FINAL CONFIG (post-diagnostic, matches reported paper numbers):
        #   tau=0.8            — selected on the validation sweep to maximise
        #                        aggregate cross-lingual validation MRR
        #                        (see docs/FIXES.md #9 and the paper Appendix)
        #   query_smoothing=0  — smoothing was found to HURT MRR on all three
        #                        languages in diagnose_medweb.sh;
        #                        do not re-enable without new evidence
        #                        (see docs/FIXES.md #10)
        #   adaptive_gamma=on  — neutral-to-mildly-positive in the same
        #                        diagnostic; safe to keep
        extra_args=(--chunk_size 512 --num_topics 20 --gamma 0.7 --tau 0.8
                    --query_smoothing 0.0 --adaptive_gamma)

        if [ "$dname" = "medweb_ja" ]; then extra_args+=(--chunk_size_ja 512); fi
        if [ "$dname" = "medweb_zh" ]; then extra_args+=(--chunk_size_zh 512); fi

        echo ""
        echo "--- ${dname} ---"

        # Lex-TM final config (gamma=0.7, tau=0.8) + RRF baseline in one run
        run_experiment "${dname} | Lex-TM + RRF" \
            "experiment_logs1/${dname}_gamma_0.7_tau_0.8_rrf.log" \
            python main_pipeline1_patched.py \
            --dataset_name "$dname" --filepath "$dpath" \
            --run_bm25 "${extra_args[@]}"

        # Standard LDA baseline (tau=1.1 disables HHI amplification)
        run_experiment "${dname} | LDA baseline" \
            "experiment_logs1/${dname}_gamma_0.7_tau_1.1.log" \
            python main_pipeline1_patched.py \
            --dataset_name "$dname" --filepath "$dpath" \
            --query_smoothing 0.0 --no_adaptive_gamma \
            --chunk_size 512 --num_topics 20 --gamma 0.7 --tau 1.1 \
            $([ "$dname" = "medweb_ja" ] && echo "--chunk_size_ja 512") \
            $([ "$dname" = "medweb_zh" ] && echo "--chunk_size_zh 512")

        # Pure dense upper bound (gamma=1.0 disables routing; tau irrelevant
        # but kept at 0.8 for log-naming consistency with the Lex-TM run)
        run_experiment "${dname} | Pure Dense" \
            "experiment_logs1/${dname}_gamma_1.0_tau_0.8.log" \
            python main_pipeline1_patched.py \
            --dataset_name "$dname" --filepath "$dpath" \
            --query_smoothing 0.0 --no_adaptive_gamma \
            --chunk_size 512 --num_topics 20 --gamma 1.0 --tau 0.8 \
            $([ "$dname" = "medweb_ja" ] && echo "--chunk_size_ja 512") \
            $([ "$dname" = "medweb_zh" ] && echo "--chunk_size_zh 512")
    done
fi

if [ "$MEDWEB_ONLY" = true ]; then
    echo ""
    echo "  --medweb-only: skipping Steps 3-6"
    echo "  Jump to RESULTS SUMMARY below."
else

# ---------------------------------------------------------------------------
# STEP 3: AVIATION — NEGATIVE RESULT (boundary condition)
# K=50 cache exists — embeddings + topicdists load instantly
# No fixes 3/4 — aviation is not a short-query corpus
# --run_bm25 adds RRF baseline
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STEP 3: Aviation (EN / ZH) — Boundary Condition"
echo "============================================================"

for dname_path in "english_aviation:$EN_AVIATION_PATH" "chinese_aviation:$ZH_AVIATION_PATH"; do
    dname="${dname_path%%:*}"
    dpath="${dname_path##*:}"

    echo ""
    echo "--- ${dname} ---"

    run_experiment "${dname} | Lex-TM + RRF" \
        "experiment_logs1/${dname}_gamma_0.7_tau_0.5_rrf.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$dpath" \
        --gamma 0.7 --tau 0.5 --num_topics 50 \
        --query_smoothing 0.0 --no_adaptive_gamma \
        --run_bm25

    run_experiment "${dname} | LDA baseline" \
        "experiment_logs1/${dname}_gamma_0.7_tau_1.1.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$dpath" \
        --gamma 0.7 --tau 1.1 --num_topics 50 \
        --query_smoothing 0.0 --no_adaptive_gamma
done

# ---------------------------------------------------------------------------
# STEP 4: CROSS-LINGUAL BENCHMARKS (MIRACL + Mr. TyDi)
# K=50 cache — fast. min_query_tokens=5 for short factoid queries.
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STEP 4: Cross-lingual Benchmarks (MIRACL ZH / Mr. TyDi JA+TH)"
echo "============================================================"

declare -A QREL_DATASETS=(
    [miracl_zh]="$MIRACL_ZH_CORPUS:$MIRACL_ZH_QUERIES:$MIRACL_ZH_QRELS"
    [mr_tydi_ja]="$MR_TYDI_JA_CORPUS:$MR_TYDI_JA_QUERIES:$MR_TYDI_JA_QRELS"
    [mr_tydi_th]="$MR_TYDI_TH_CORPUS:$MR_TYDI_TH_QUERIES:$MR_TYDI_TH_QRELS"
)

for dname in miracl_zh mr_tydi_ja mr_tydi_th; do
    IFS=':' read -r corpus queries qrels <<< "${QREL_DATASETS[$dname]}"

    if [ ! -f "$corpus" ]; then
        echo "  WARNING: $dname corpus not found — skipping."
        FAILED_RUNS+=("$dname — files not found")
        continue
    fi

    echo ""
    echo "--- ${dname} ---"

    run_experiment "${dname} | Lex-TM + RRF" \
        "experiment_logs1/${dname}_gamma_0.7_tau_0.5_rrf.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$corpus" \
        --queries_path "$queries" --qrels_path "$qrels" \
        --gamma 0.7 --tau 0.5 --num_topics 50 \
        --min_query_tokens 5 \
        --query_smoothing 0.0 --no_adaptive_gamma \
        --run_bm25

    run_experiment "${dname} | LDA baseline" \
        "experiment_logs1/${dname}_gamma_0.7_tau_1.1.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$corpus" \
        --queries_path "$queries" --qrels_path "$qrels" \
        --gamma 0.7 --tau 1.1 --num_topics 50 \
        --min_query_tokens 5 \
        --query_smoothing 0.0 --no_adaptive_gamma
done

# ---------------------------------------------------------------------------
# STEP 5: GENERAL-DOMAIN CONTROLS (THUCNews + Sogou)
# K=50 cache — fast. No fixes 3/4.
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STEP 5: General-Domain Controls (THUCNews + Sogou)"
echo "============================================================"

for dname_path in "thucnews:$THUCNEWS_PATH" "sogou:$SOGOU_PATH"; do
    dname="${dname_path%%:*}"
    dpath="${dname_path##*:}"

    echo ""
    echo "--- ${dname} ---"

    run_experiment "${dname} | Lex-TM + RRF" \
        "experiment_logs1/${dname}_gamma_0.7_tau_0.5_rrf.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$dpath" \
        --gamma 0.7 --tau 0.5 --num_topics 50 \
        --query_smoothing 0.0 --no_adaptive_gamma \
        --run_bm25

    run_experiment "${dname} | LDA baseline" \
        "experiment_logs1/${dname}_gamma_0.7_tau_1.1.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$dpath" \
        --gamma 0.7 --tau 1.1 --num_topics 50 \
        --query_smoothing 0.0 --no_adaptive_gamma
done

# ---------------------------------------------------------------------------
# STEP 6: BEIR BOUNDARY CONDITIONS (NFCorpus + TREC-COVID)
# K=50 cache — fast. Expected to show degradation (boundary condition).
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STEP 6: BEIR Boundary Conditions (NFCorpus + TREC-COVID)"
echo "============================================================"

declare -A BEIR_DATASETS=(
    [nfcorpus]="$NFCORPUS_CORPUS:$NFCORPUS_QUERIES:$NFCORPUS_QRELS"
    [trec_covid]="$TREC_COVID_CORPUS:$TREC_COVID_QUERIES:$TREC_COVID_QRELS"
)

for dname in nfcorpus trec_covid; do
    IFS=':' read -r corpus queries qrels <<< "${BEIR_DATASETS[$dname]}"

    if [ ! -f "$corpus" ]; then
        echo "  WARNING: $dname corpus not found — skipping."
        FAILED_RUNS+=("$dname — files not found")
        continue
    fi

    echo ""
    echo "--- ${dname} ---"

    run_experiment "${dname} | Lex-TM + RRF" \
        "experiment_logs1/${dname}_gamma_0.7_tau_0.5_rrf.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$corpus" \
        --queries_path "$queries" --qrels_path "$qrels" \
        --gamma 0.7 --tau 0.5 --num_topics 50 \
        --query_smoothing 0.0 --no_adaptive_gamma \
        --run_bm25

    run_experiment "${dname} | LDA baseline" \
        "experiment_logs1/${dname}_gamma_0.7_tau_1.1.log" \
        python main_pipeline1_patched.py \
        --dataset_name "$dname" --filepath "$corpus" \
        --queries_path "$queries" --qrels_path "$qrels" \
        --gamma 0.7 --tau 1.1 --num_topics 50 \
        --query_smoothing 0.0 --no_adaptive_gamma
done

fi  # end of --medweb-only skip

# ---------------------------------------------------------------------------
# RESULTS SUMMARY
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  RESULTS SUMMARY"
echo "============================================================"
printf "%-22s  %-8s  %-8s  %-8s\n" "Dataset" "Lex-TM" "LDA" "RRF"
echo "------------------------------------------------------"

all_datasets=(medweb_en medweb_ja medweb_zh
              english_aviation chinese_aviation
              miracl_zh mr_tydi_ja mr_tydi_th
              thucnews sogou nfcorpus trec_covid)

# Primary tau used for the Lex-TM+RRF log filename, per dataset.
# MedWeb uses tau=0.8 (validated in the diagnostic — see docs/FIXES.md #9).
# All other datasets remain at tau=0.5 (never re-validated at 0.8).
declare -A PRIMARY_TAU=(
    [medweb_en]="0.8" [medweb_ja]="0.8" [medweb_zh]="0.8"
    [english_aviation]="0.5" [chinese_aviation]="0.5"
    [miracl_zh]="0.5" [mr_tydi_ja]="0.5" [mr_tydi_th]="0.5"
    [thucnews]="0.5" [sogou]="0.5" [nfcorpus]="0.5" [trec_covid]="0.5"
)

for dname in "${all_datasets[@]}"; do
    ptau="${PRIMARY_TAU[$dname]}"
    rrf_log="experiment_logs1/${dname}_gamma_0.7_tau_${ptau}_rrf.log"
    lda_log="experiment_logs1/${dname}_gamma_0.7_tau_1.1.log"

    # First MRR line = Lex-TM (evaluate_retrieval runs first)
    # RRF MRR follows the "BM25+Dense RRF Evaluation" header
    lextm_mrr=$(grep "^MRR:" "$rrf_log" 2>/dev/null | head -1 | awk '{print $2}')
    lda_mrr=$(  grep "^MRR:" "$lda_log" 2>/dev/null | head -1 | awk '{print $2}')
    rrf_mrr=$(  grep -A3 "BM25+Dense RRF Evaluation" "$rrf_log" 2>/dev/null \
                | grep "^MRR:" | head -1 | awk '{print $2}')

    printf "%-22s  %-8s  %-8s  %-8s\n" \
        "$dname" "${lextm_mrr:--}" "${lda_mrr:--}" "${rrf_mrr:--}"
done

echo "============================================================"

if [ ${#FAILED_RUNS[@]} -eq 0 ]; then
    echo "  All experiments completed successfully."
else
    echo "  WARNING: ${#FAILED_RUNS[@]} run(s) failed:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "    - $run"
    done
    echo "  Check the .log files in experiment_logs1/ for details."
fi
echo "============================================================"
