#!/bin/bash
# =============================================================================
# Lex-TM Full Evaluation Suite — AACL 2026
#
# Dataset roles
# -------------
# Section 1  MedWeb (en/ja/zh)         PRIMARY specialised domain
# Section 2  Aviation (en/zh)           NEGATIVE RESULT — passenger reviews
# Section 3  MIRACL + Mr. TyDi          Cross-lingual benchmarks, human qrels
# Section 4  THUCNews test / Sogou      General-domain controls
# Section 5  LLM-as-a-Judge            Local Qwen 2.5 32B via Ollama
#
# MedWeb chunk sizes
# ------------------
# MedWeb texts are short social media posts (~12-15 tokens). Use:
#   --chunk_size 20 --chunk_size_ja 30 --chunk_size_zh 30 --num_topics 20
#
# LLM Judge prerequisite
# -----------------------
# Pull the model once before running:
#   ollama run qwen2.5:32b
#   (Type /bye to exit — the server stays running in the background)
#
# Usage
# -----
#   bash run_experiment.sh
#
# Logs: experiment_logs1/<dataset>_<config>.log
# =============================================================================

set -u

mkdir -p experiment_logs1

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
MEDWEB_PATH="../data/SOTA/ntcir13_MedWeb_TestCollection/medweb_rag_all_fixed.csv"
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

THUCNEWS_TEST_PATH="../data/SOTA/thucnews_test.csv"
THUCNEWS_PATH="../data/SOTA/thucnews1.csv"
SOGOU_PATH="../data/SOTA/sogou_news1.csv"

# General-domain corpus cap. Override at call time:
#   MAX_DOCS_GENERAL=5000 bash run_experiment.sh
MAX_DOCS_GENERAL="${MAX_DOCS_GENERAL:-0}"#100000
FAILED_RUNS=()

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run_experiment() {
    local label="$1"
    local log_file="$2"
    shift 2

    echo "  Running : $label"
    if "$@" > "$log_file" 2>&1; then
        echo "  ✓  Done   -> $log_file"
    else
        echo "  ✗  FAILED (exit $?) -> see $log_file"
        FAILED_RUNS+=("$label")
    fi
}

echo ""
echo "============================================================"
echo "  Lex-TM Evaluation Suite — AACL 2026"
#echo "  General-domain cap: MAX_DOCS_GENERAL=${MAX_DOCS_GENERAL}"
echo "============================================================"

# =============================================================================
# SECTION 1: MEDWEB — PRIMARY SPECIALISED DOMAIN
#
# For loop over three parallel language partitions (en/ja/zh).
# Each language runs: LDA baseline, Lex-TM optimal, tau ablation,
# gamma sweep low, pure dense.
#
# FIX (vs the original version): $extra_args was unquoted which caused
# word-splitting on "--chunk_size_zh 30" — bash split it at the space
# and Python received "--chunk_size_zh" and "30" as separate positional
# arguments rather than one flag-value pair. Fixed by using an array
# instead of a string variable.
# =============================================================================
echo ""
echo "--- MedWeb Parallel Ablation (EN / JA / ZH) ---"

declare -A MEDWEB_PATHS=(
    [medweb_en]="$MEDWEB_EN_PATH"
    [medweb_ja]="$MEDWEB_JA_PATH"
    [medweb_zh]="$MEDWEB_ZH_PATH"
)

for dname in medweb_en medweb_ja medweb_zh; do
    dpath="${MEDWEB_PATHS[$dname]}"

    # Build extra args as an array — safe against word-splitting
    # chunk_size=512: MedWeb posts average 12-15 tokens. At chunk_size=20,
    # many posts fall below min_chunk_len and are discarded entirely.
    # Setting chunk_size=512 (larger than any post) keeps each document
    # as one chunk — correct granularity for short social media posts.
    extra_args=(--chunk_size 512 --num_topics 20)
    if [ "$dname" = "medweb_ja" ]; then
        extra_args+=(--chunk_size_ja 512)
    fi
    if [ "$dname" = "medweb_zh" ]; then
        extra_args+=(--chunk_size_zh 512)
    fi

    # LDA baseline (tau=1.1 disables all HHI amplification)
    run_experiment "$dname | LDA baseline | gamma=0.7 tau=1.1" \
        "experiment_logs1/${dname}_gamma_0.7_tau_1.1.log" \
        python main_pipeline1.py --dataset_name "$dname" --filepath "$dpath" \
        --gamma 0.7 --tau 1.1 "${extra_args[@]}"

    # Lex-TM primary (paper optimal: gamma=0.7 tau=0.5)
    run_experiment "$dname | Lex-TM | gamma=0.7 tau=0.5" \
        "experiment_logs1/${dname}_gamma_0.7_tau_0.5.log" \
        python main_pipeline1.py --dataset_name "$dname" --filepath "$dpath" \
        --gamma 0.7 --tau 0.5 "${extra_args[@]}"

    # Tau ablation
    run_experiment "$dname | ablation | gamma=0.7 tau=0.8" \
        "experiment_logs1/${dname}_gamma_0.7_tau_0.8.log" \
        python main_pipeline1.py --dataset_name "$dname" --filepath "$dpath" \
        --gamma 0.7 --tau 0.8 "${extra_args[@]}"

    # Gamma sweep — light penalty
    run_experiment "$dname | Lex-TM | gamma=0.9 tau=0.5" \
        "experiment_logs1/${dname}_gamma_0.9_tau_0.5.log" \
        python main_pipeline1.py --dataset_name "$dname" --filepath "$dpath" \
        --gamma 0.9 --tau 0.5 "${extra_args[@]}"

    # Pure dense (gamma=1.0 — upper bound comparison)
    run_experiment "$dname | pure dense | gamma=1.0" \
        "experiment_logs1/${dname}_gamma_1.0_tau_0.5.log" \
        python main_pipeline1.py --dataset_name "$dname" --filepath "$dpath" \
        --gamma 1.0 --tau 0.5 "${extra_args[@]}"

    # HHI scope ablation on zh only (most AACL-relevant language)
    if [ "$dname" = "medweb_zh" ] || [ "$dname" = "medweb_ja" ]; then
        run_experiment "$dname | ablation | HHI scope=document" \
            "experiment_logs1/${dname}_gamma_0.7_tau_0.5_hhi_document.log" \
            python main_pipeline1.py --dataset_name "$dname" --filepath "$dpath" \
            --gamma 0.7 --tau 0.5 --hhi_scope document "${extra_args[@]}"

        run_experiment "$dname | ablation | HHI scope=chunk" \
            "experiment_logs1/${dname}_gamma_0.7_tau_0.5_hhi_chunk.log" \
            python main_pipeline1.py --dataset_name "$dname" --filepath "$dpath" \
            --gamma 0.7 --tau 0.5 --hhi_scope chunk "${extra_args[@]}"
    fi

done

echo "--- MedWeb complete (17 runs across en/ja/zh) ---"

echo ""
echo "--- MFCorpus Parallel Ablation (EN) ---"

python main_pipeline.py --dataset_name "nfcorpus" \
    --filepath "../data/beir/nfcorpus/corpus.jsonl" \
    --queries_path "../data/beir/nfcorpus/queries.jsonl" \
    --qrels_path "../data/beir/nfcorpus/qrels/test.tsv" \
    --gamma 0.7 --tau 0.5 --num_topics 50

python main_pipeline.py --dataset_name "nfcorpus" \
    --filepath "../data/beir/nfcorpus/corpus.jsonl" \
    --queries_path "../data/beir/nfcorpus/queries.jsonl" \
    --qrels_path "../data/beir/nfcorpus/qrels/test.tsv" \
    --gamma 0.7 --tau 1.1 --num_topics 50

python main_pipeline.py --dataset_name "nfcorpus" \
    --filepath "../data/beir/nfcorpus/corpus.jsonl" \
    --queries_path "../data/beir/nfcorpus/queries.jsonl" \
    --qrels_path "../data/beir/nfcorpus/qrels/test.tsv" \
    --gamma 1.0 --tau 0.5 --num_topics 50


echo "--- NFCorpus complete (3 runs en) ---"

# =============================================================================
# SECTION 2: AVIATION — NEGATIVE RESULT
# Passenger reviews lack semantic collapse. Retained as honest scope boundary.
# Only 3 runs per language: baseline, Lex-TM, pure dense.
# =============================================================================
echo ""
echo "--- English Aviation (negative result — passenger reviews) ---"

run_experiment "en_aviation | LDA baseline | gamma=0.7 tau=1.1" \
    "experiment_logs1/english_aviation_gamma_0.7_tau_1.1.log" \
    python main_pipeline1.py \
    --dataset_name "english_aviation" --filepath "$EN_AVIATION_PATH" \
    --gamma 0.7 --tau 1.1 --num_topics 50

run_experiment "en_aviation | Lex-TM | gamma=0.7 tau=0.5" \
    "experiment_logs1/english_aviation_gamma_0.7_tau_0.5.log" \
    python main_pipeline1.py \
    --dataset_name "english_aviation" --filepath "$EN_AVIATION_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 50

run_experiment "en_aviation | pure dense | gamma=1.0" \
    "experiment_logs1/english_aviation_gamma_1.0_tau_0.5.log" \
    python main_pipeline1.py \
    --dataset_name "english_aviation" --filepath "$EN_AVIATION_PATH" \
    --gamma 1.0 --tau 0.5 --num_topics 50

echo "--- Chinese Aviation (negative result — passenger reviews) ---"

run_experiment "zh_aviation | LDA baseline | gamma=0.7 tau=1.1" \
    "experiment_logs1/chinese_aviation_gamma_0.7_tau_1.1.log" \
    python main_pipeline1.py \
    --dataset_name "chinese_aviation" --filepath "$ZH_AVIATION_PATH" \
    --gamma 0.7 --tau 1.1 --num_topics 50

run_experiment "zh_aviation | Lex-TM | gamma=0.7 tau=0.5" \
    "experiment_logs1/chinese_aviation_gamma_0.7_tau_0.5.log" \
    python main_pipeline1.py \
    --dataset_name "chinese_aviation" --filepath "$ZH_AVIATION_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 50

run_experiment "zh_aviation | pure dense | gamma=1.0" \
    "experiment_logs1/chinese_aviation_gamma_1.0_tau_0.5.log" \
    python main_pipeline1.py \
    --dataset_name "chinese_aviation" --filepath "$ZH_AVIATION_PATH" \
    --gamma 1.0 --tau 0.5 --num_topics 50

echo "--- Aviation complete (6 runs) ---"

# =============================================================================
# SECTION 3: CROSS-LINGUAL BENCHMARKS (MIRACL + Mr. TyDi)
#
# FIX (vs the original version): LDA baselines were missing — only Lex-TM was run.
# Without the baseline you cannot compute the MRR delta for these datasets.
# Both conditions are now run for every benchmark.
# =============================================================================
echo ""
echo "--- MIRACL Chinese (100K-passage Wikipedia, human qrels) ---"

if [ ! -f "$MIRACL_ZH_CORPUS" ]; then
    echo "  WARNING: MIRACL files not found. Skipping."
    echo "  Run: python download_miracl_chinese.py --output_dir ../data/miracl_zh"
    FAILED_RUNS+=("miracl_zh — files not found")
else
    run_experiment "miracl_zh | LDA baseline | gamma=0.7 tau=1.1" \
        "experiment_logs1/miracl_zh_gamma_0.7_tau_1.1.log" \
        python main_pipeline1.py \
        --dataset_name "miracl_zh" --filepath "$MIRACL_ZH_CORPUS" \
        --queries_path "$MIRACL_ZH_QUERIES" --qrels_path "$MIRACL_ZH_QRELS" \
        --gamma 0.7 --tau 1.1 --num_topics 50

    run_experiment "miracl_zh | Lex-TM | gamma=0.7 tau=0.5" \
        "experiment_logs1/miracl_zh_gamma_0.7_tau_0.5.log" \
        python main_pipeline1.py \
        --dataset_name "miracl_zh" --filepath "$MIRACL_ZH_CORPUS" \
        --queries_path "$MIRACL_ZH_QUERIES" --qrels_path "$MIRACL_ZH_QRELS" \
        --gamma 0.7 --tau 0.5 --num_topics 50

    run_experiment "miracl_zh | pure dense | gamma=1.0 tau=0.5" \
        "experiment_logs1/miracl_zh_gamma_1.0_tau_0.5.log" \
        python main_pipeline1.py \
        --dataset_name "miracl_zh" --filepath "$MIRACL_ZH_CORPUS" \
        --queries_path "$MIRACL_ZH_QUERIES" --qrels_path "$MIRACL_ZH_QRELS" \
        --gamma 1.0 --tau 0.5 --num_topics 50
fi

echo ""
echo "--- Mr. TyDi Japanese ---"

if [ ! -f "$MR_TYDI_JA_CORPUS" ]; then
    echo "  WARNING: Mr. TyDi Japanese files not found. Skipping."
    echo "  Run: python download_mr_tydi.py --language japanese --output_dir ../data/mr_tydi_ja"
    FAILED_RUNS+=("mr_tydi_ja — files not found")
else
    run_experiment "mr_tydi_ja | LDA baseline | gamma=0.7 tau=1.1" \
        "experiment_logs1/mr_tydi_ja_gamma_0.7_tau_1.1.log" \
        python main_pipeline1.py \
        --dataset_name "mr_tydi_ja" --filepath "$MR_TYDI_JA_CORPUS" \
        --queries_path "$MR_TYDI_JA_QUERIES" --qrels_path "$MR_TYDI_JA_QRELS" \
        --gamma 0.7 --tau 1.1 --num_topics 50

    run_experiment "mr_tydi_ja | Lex-TM | gamma=0.7 tau=0.5" \
        "experiment_logs1/mr_tydi_ja_gamma_0.7_tau_0.5.log" \
        python main_pipeline1.py \
        --dataset_name "mr_tydi_ja" --filepath "$MR_TYDI_JA_CORPUS" \
        --queries_path "$MR_TYDI_JA_QUERIES" --qrels_path "$MR_TYDI_JA_QRELS" \
        --gamma 0.7 --tau 0.5 --num_topics 50

    run_experiment "mr_tydi_ja | pure dense | gamma=1.0 tau=0.5" \
        "experiment_logs1/mr_tydi_ja_gamma_1.0_tau_0.5.log" \
        python main_pipeline1.py \
        --dataset_name "mr_tydi_ja" --filepath "$MR_TYDI_JA_CORPUS" \
        --queries_path "$MR_TYDI_JA_QUERIES" --qrels_path "$MR_TYDI_JA_QRELS" \
        --gamma 1.0 --tau 0.5 --num_topics 50
fi

echo ""
echo "--- Mr. TyDi Thai ---"

if [ ! -f "$MR_TYDI_TH_CORPUS" ]; then
    echo "  WARNING: Mr. TyDi Thai files not found. Skipping."
    echo "  Run: python download_mr_tydi.py --language thai --output_dir ../data/mr_tydi_th"
    FAILED_RUNS+=("mr_tydi_th — files not found")
else
    run_experiment "mr_tydi_th | LDA baseline | gamma=0.7 tau=1.1" \
        "experiment_logs1/mr_tydi_th_gamma_0.7_tau_1.1.log" \
        python main_pipeline1.py \
        --dataset_name "mr_tydi_th" --filepath "$MR_TYDI_TH_CORPUS" \
        --queries_path "$MR_TYDI_TH_QUERIES" --qrels_path "$MR_TYDI_TH_QRELS" \
        --gamma 0.7 --tau 1.1 --num_topics 50

    run_experiment "mr_tydi_th | Lex-TM | gamma=0.7 tau=0.5" \
        "experiment_logs1/mr_tydi_th_gamma_0.7_tau_0.5.log" \
        python main_pipeline1.py \
        --dataset_name "mr_tydi_th" --filepath "$MR_TYDI_TH_CORPUS" \
        --queries_path "$MR_TYDI_TH_QUERIES" --qrels_path "$MR_TYDI_TH_QRELS" \
        --gamma 0.7 --tau 0.5 --num_topics 50

    run_experiment "mr_tydi_th | pure dense | gamma=1.0 tau=0.5" \
        "experiment_logs1/mr_tydi_th_gamma_1.0_tau_0.5.log" \
        python main_pipeline1.py \
        --dataset_name "mr_tydi_th" --filepath "$MR_TYDI_TH_CORPUS" \
        --queries_path "$MR_TYDI_TH_QUERIES" --qrels_path "$MR_TYDI_TH_QRELS" \
        --gamma 1.0 --tau 0.5 --num_topics 50
fi

echo "--- Cross-lingual benchmarks complete ---"

# =============================================================================
# SECTION 4: GENERAL-DOMAIN CONTROLS (THUCNews + Sogou)
# Expected: near-zero delta confirms Lex-TM does not hurt non-specialised text.
# =============================================================================
echo ""
echo "--- THUCNews (20%) test split (132699 docs) ---"

run_experiment "thucnews | LDA baseline | gamma=0.7 tau=1.1" \
    "experiment_logs1/thucnews_test_gamma_0.7_tau_1.1.log" \
    python main_pipeline1.py \
    --dataset_name "thucnews" --filepath "$THUCNEWS_TEST_PATH" \
    --gamma 0.7 --tau 1.1 --num_topics 50

run_experiment "thucnews | Lex-TM | gamma=0.7 tau=0.5" \
    "experiment_logs1/thucnews_test_gamma_0.7_tau_0.5.log" \
    python main_pipeline1.py \
    --dataset_name "thucnews" --filepath "$THUCNEWS_TEST_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 50

echo ""
echo "--- THUCNews test split (~5K docs) ---"

run_experiment "thucnews_test | LDA baseline | gamma=0.7 tau=1.1" \
    "experiment_logs1/thucnews_test_gamma_0.7_tau_1.1.log" \
    python main_pipeline1.py \
    --dataset_name "thucnews" --filepath "$THUCNEWS_PATH" \
    --gamma 0.7 --tau 1.1 --num_topics 50 \
    --max_docs 5000

run_experiment "thucnews_test | Lex-TM | gamma=0.7 tau=0.5" \
    "experiment_logs1/thucnews_test_gamma_0.7_tau_0.5.log" \
    python main_pipeline1.py \
    --dataset_name "thucnews" --filepath "$THUCNEWS_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 50 \
    --max_docs 5000

# =============================================================================
# 4a. THUCNEWS — general-domain Chinese sanity check (2 runs)
#
# Purpose: verify Lex-TM does not degrade retrieval on non-specialised text.
# Corpus is capped at MAX_DOCS_GENERAL (default 100K) so runtime is tractable.
# Full THUCNews (~740K articles) takes ~5 days per run at this corpus size.
# The cap and seed are fixed and must be disclosed in the paper.
#
# Only gamma=0.7 is tested — the optimal value from the aviation ablation.
# A gamma sweep on the general-domain datasets is not needed because these
# rows are sanity checks, not ablation targets.
# =============================================================================
echo ""
echo "--- THUCNews (capped at ${MAX_DOCS_GENERAL} docs) ---"

run_experiment "thucnews | LDA baseline | gamma=0.7 tau=1.1 | max_docs=${MAX_DOCS_GENERAL}" \
    "experiment_logs1/thucnews_gamma_0.7_tau_1.1_maxdocs${MAX_DOCS_GENERAL}.log" \
    python main_pipeline1.py \
    --dataset_name "thucnews" --filepath "$THUCNEWS_PATH" \
    --gamma 0.7 --tau 1.1 --num_topics 50 \
    --max_docs "$MAX_DOCS_GENERAL"

run_experiment "thucnews | Lex-TM | gamma=0.7 tau=0.5 | max_docs=${MAX_DOCS_GENERAL}" \
    "experiment_logs1/thucnews_gamma_0.7_tau_0.5_maxdocs${MAX_DOCS_GENERAL}.log" \
    python main_pipeline1.py \
    --dataset_name "thucnews" --filepath "$THUCNEWS_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 50
    --max_docs "$MAX_DOCS_GENERAL"

echo "--- THUCNews complete (2 runs) ---"

echo ""
echo "--- Sogou News (10,497 docs) ---"

run_experiment "sogou | LDA baseline | gamma=0.7 tau=1.1" \
    "experiment_logs1/sogou_gamma_0.7_tau_1.1.log" \
    python main_pipeline1.py \
    --dataset_name "sogou" --filepath "$SOGOU_PATH" \
    --gamma 0.7 --tau 1.1 --num_topics 50

run_experiment "sogou | Lex-TM | gamma=0.7 tau=0.5" \
    "experiment_logs1/sogou_gamma_0.7_tau_0.5.log" \
    python main_pipeline1.py \
    --dataset_name "sogou" --filepath "$SOGOU_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 50

echo "--- Sogou News complete (2 runs) ---"
echo "--- General-domain controls complete (4 runs) ---"

# =============================================================================
# SECTION 5: LLM-AS-A-JUDGE TOPIC EVALUATION (LOCAL QWEN 2.5 32B)
#
# Uses local Ollama server instead of GPT-4o API (from the original version).
# Prerequisite: ollama run qwen2.5:32b  (pull once, server stays running)
# A dummy API key is set — the OpenAI client points to localhost:11434.
# =============================================================================
echo ""
echo "--- LLM-as-a-Judge Topic Evaluation (Local Qwen 2.5 32B) ---"

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  WARNING: Ollama server not running on localhost:11434. Skipping LLM judge."
    echo "  Start with: ollama run qwen2.5:32b"
    FAILED_RUNS+=("LLMjudge.py — Ollama not running")
else
    export OPENAI_API_KEY="ollama-local"
    echo "  Running LLM judge (local Qwen 2.5 32B)..."
    if python LLMjudge.py > experiment_logs1/llm_judge.log 2>&1; then
        echo "  ✓  LLM judge complete -> experiment_logs1/llm_judge.log"
        echo "     Full grades: master_llm_grades.json"
    else
        echo "  ✗  LLM judge FAILED -> see experiment_logs1/llm_judge.log"
        FAILED_RUNS+=("LLMjudge.py")
    fi
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo "  Run summary"
echo "  MedWeb (en+ja+zh)         : 17 runs"
echo "  Aviation (en+zh)           :  6 runs  (negative result)"
echo "  MIRACL + Mr. TyDi (ja+th) :  6 runs  (if files present)"
echo "  THUCNews + Sogou           :  4 runs"
echo "  Total                      : 33 runs"
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

echo "  Results: experiment_logs1/"
echo "============================================================"
