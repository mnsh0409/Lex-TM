set -u

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



echo ""
echo "============================================================"
echo "  Lex-TM Evaluation Suite — AACL 2026" - validation
echo "============================================================"

# =============================================================================
# SECTION 1: MEDWEB — PRIMARY SPECIALISED DOMAIN
# =============================================================================
echo ""
echo "--- MedWeb Parallel Ablation (EN / JA / ZH) ---"


echo ""
echo "--- MedWeb English ---"

# Sweep on validation set to pick hyperparameters
python main_pipeline_valid.py --dataset_name "medweb_en" \
    --filepath "$MEDWEB_EN_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 20 --chunk_size 512 \
    --val_fraction 0.2   # prints both val and test MRR

python main_pipeline_valid.py --dataset_name "medweb_en" \
    --filepath "$MEDWEB_EN_PATH" \
    --gamma 0.7 --tau 0.8 --num_topics 20 --chunk_size 512 \
    --val_fraction 0.2   # compare val MRR here

# After selecting best tau from val MRR — run final with --run_bm25
python main_pipeline_valid.py --dataset_name "medweb_en" \
    --filepath "$MEDWEB_EN_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 20 --chunk_size 512 \
    --val_fraction 0.2 --run_bm25   # final: report test MRR


echo ""
echo "--- MedWeb Chinese ---"

# Sweep on validation set to pick hyperparameters
python main_pipeline_valid.py --dataset_name "medweb_zh" \
    --filepath "$MEDWEB_ZH_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 20 --chunk_size 512 \
    --val_fraction 0.2   # prints both val and test MRR

python main_pipeline_valid.py --dataset_name "medweb_zh" \
    --filepath "$MEDWEB_ZH_PATH" \
    --gamma 0.7 --tau 0.8 --num_topics 20 --chunk_size 512 \
    --val_fraction 0.2   # compare val MRR here

# After selecting best tau from val MRR — run final with --run_bm25
python main_pipeline_valid.py --dataset_name "medweb_zh" \
    --filepath "$MEDWEB_ZH_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 20 --chunk_size 512 \
    --val_fraction 0.2 --run_bm25   # final: report test MRR


echo ""
echo "--- MedWeb Japanese ---"

# Sweep on validation set to pick hyperparameters
python main_pipeline_valid.py --dataset_name "medweb_ja" \
    --filepath "$MEDWEB_JA_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 20 --chunk_size 512 \
    --val_fraction 0.2   # prints both val and test MRR

python main_pipeline_valid.py --dataset_name "medweb_ja" \
    --filepath "$MEDWEB_JA_PATH" \
    --gamma 0.7 --tau 0.8 --num_topics 20 --chunk_size 512 \
    --val_fraction 0.2   # compare val MRR here

# After selecting best tau from val MRR — run final with --run_bm25
python main_pipeline_valid.py --dataset_name "medweb_ja" \
    --filepath "$MEDWEB_JA_PATH" \
    --gamma 0.7 --tau 0.5 --num_topics 20 --chunk_size 512 \
    --val_fraction 0.2 --run_bm25   # final: report test MRR


echo "--- MedWeb validation complete (9 runs across en/ja/zh) ---"
