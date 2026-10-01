#!/usr/bin/env bash
# copy_cache.sh
# =============================================================================
# Reuses BGE-m3 embedding caches from the original pipeline (experiment_logs1/)
# in the validation pipeline (experiment_logs/).
#
# WHY: The two pipelines use different log directories:
#   Original   : experiment_logs1/cache_{dataset}_tau{tau}_K{K}_embeddings.npy
#   Validation : experiment_logs/cache_{dataset}_tau{tau}_K{K}_embeddings.npy
#
# The embeddings are IDENTICAL (same model, same corpus, same chunk_size).
# Copying avoids redundant BGE-m3 re-encoding (15-30 min per dataset on GPU).
#
# What NOT to copy: _topicdists.npy files depend on K and tau.
#   K=50 topicdists CANNOT be reused for K=20 runs — different array shapes.
#   Only copy embeddings; let topic distributions recompute.
#
# Usage
# -----
#   bash copy_cache.sh               # copies all datasets listed below
#   bash copy_cache.sh medweb_en     # copies one specific dataset
#
# Safe to re-run: cp -n (no-clobber) skips files that already exist.
# =============================================================================

set -euo pipefail

SRC_DIR="experiment_logs1"
DST_DIR="experiment_logs"

mkdir -p "$DST_DIR"

# Datasets and tau/K combos that exist in experiment_logs1/
# Adjust this list to match what you actually have cached.
DATASETS=(
    "medweb_en"
    "medweb_ja"
    "medweb_zh"
    "miracl_zh"
    "mr_tydi_ja"
    "mr_tydi_th"
    "english_aviation"
    "chinese_aviation"
    "thucnews"
    "sogou"
    "trec_covid"
    "nfcorpus"
)

TAUS=("0.5" "0.8" "1.1")
KS=("20" "50")

# If a specific dataset is passed as argument, only copy that one
if [ $# -ge 1 ]; then
    DATASETS=("$1")
fi

COPIED=0
SKIPPED=0
MISSING=0

for dataset in "${DATASETS[@]}"; do
    for tau in "${TAUS[@]}"; do
        for k in "${KS[@]}"; do
            src="${SRC_DIR}/cache_${dataset}_tau${tau}_K${k}_embeddings.npy"
            dst="${DST_DIR}/cache_${dataset}_tau${tau}_K${k}_embeddings.npy"

            if [ ! -f "$src" ]; then
                # Not every tau/K combo exists — that's expected
                ((MISSING++)) || true
                continue
            fi

            if [ -f "$dst" ]; then
                echo "  SKIP (exists): $dst"
                ((SKIPPED++)) || true
            else
                cp "$src" "$dst"
                echo "  COPY: $src -> $dst"
                ((COPIED++)) || true
            fi
        done
    done
done

echo ""
echo "Done. Copied: ${COPIED}  |  Skipped (already exists): ${SKIPPED}  |  Not found in src: ${MISSING}"
echo ""
echo "NOTE: _topicdists.npy files are NOT copied."
echo "  K=50 topic distributions cannot be reused for K=20 runs — different shapes."
echo "  They will recompute automatically on first run (Gibbs sampler, CPU-bound)."
