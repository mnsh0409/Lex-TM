#!/usr/bin/env bash
# =============================================================================
# run_camera_ready.sh - every experiment of the camera-ready paper in one run.
#
#   export LEXTM_DATA=/path/to/data   # holds ntcir13_MedWeb_TestCollection/ and beir/
#   bash scripts/run_camera_ready.sh  # optional: OUT=/some/dir bash scripts/run_camera_ready.sh
#
# Writes medweb_results.json, beir_<name>_results.json, per_query/*.json,
# one .log per block and results_generated.tex to $OUT
# (default: results/camera_ready_rerun). The published outputs are in
# results/camera_ready/.
#
# Times of our run on one RTX 4090: MedWeb ~5 min (cached document embeddings),
# FiQA-2018 ~1.5 h, ChatDoctor ~20 min, CmedqaRetrieval (full corpus, with the
# export) ~2.3 h. Blocks are independent: if one fails, the others still run.
# =============================================================================
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPE="$ROOT/pipelines/v3_patched"
DATA="${LEXTM_DATA:?set LEXTM_DATA to the directory that holds ntcir13_MedWeb_TestCollection/ and beir/}"
OUT="${OUT:-$ROOT/results/camera_ready_rerun}"
PY="python $PIPE/camera_ready_extras.py"
mkdir -p "$OUT"
cd "$PIPE" || exit 1

python -c "import sentence_transformers, numba, jieba, fugashi, ipadic, scipy, pandas; print('environment OK')" || exit 1
python -c "from nltk.tokenize import word_tokenize; word_tokenize('A quick check.')" \
  || { echo "NLTK punkt is missing: python -m nltk.downloader punkt punkt_tab stopwords wordnet"; exit 1; }
python -c "import FlagEmbedding" 2>/dev/null && echo "FlagEmbedding present" \
  || echo "FlagEmbedding not installed: the BGE-M3 sparse and multi-vector heads are rebuilt from transformers"

# 1) MedWeb EN/JA/ZH with BGE-M3 and Qwen3-Embedding-0.6B: BM25, dense, RRF,
#    Lex-TM, LDA routing, gamma/tau sweeps, BGE-M3 sparse and multi-vector,
#    SPLADE++ (EN), paired tests, Gibbs seeds 42-46, query-type breakdown,
#    Figure 1 candidates, label-free control.
$PY medweb --medweb_dir "$DATA/ntcir13_MedWeb_TestCollection" --out "$OUT" \
    --encoders bge-m3,qwen3-0.6b 2>&1 | tee "$OUT/medweb.log"

# 2) FiQA-2018 (finance), full corpus, downloaded from BEIR on first use.
$PY beir --dataset fiqa --data_dir "$DATA/beir/fiqa" --split test --lang en \
    --download --out "$OUT" 2>&1 | tee "$OUT/fiqa.log"

# 3) ChatDoctorRetrieval (English medical Q&A).
[ -f "$DATA/beir/chatdoctor/corpus.jsonl" ] || \
    python export_chatdoctor_to_beir.py --out_dir "$DATA/beir/chatdoctor"
$PY beir --dataset chatdoctor --data_dir "$DATA/beir/chatdoctor" \
    --split test --lang en --out "$OUT" 2>&1 | tee "$OUT/chatdoctor.log"

# 4) CmedqaRetrieval (Chinese medical QA), full corpus, dev split;
#    multi-vector skipped (about 4,000 queries over 100,000 answers).
$PY export_cmedqa --out_dir "$DATA/beir/cmedqa" --n_docs 0 2>&1 | tee "$OUT/cmedqa_export.log"
$PY beir --dataset cmedqa --data_dir "$DATA/beir/cmedqa" --split dev --lang zh \
    --no_multivector --out "$OUT" 2>&1 | tee "$OUT/cmedqa.log"

# Macro file read by the paper. FIG1_EN=1 selects the English case shown in Figure 1.
FIG1_EN=1 $PY macros --out "$OUT"
date > "$OUT/DONE.txt"
echo "All blocks finished -> $OUT"
