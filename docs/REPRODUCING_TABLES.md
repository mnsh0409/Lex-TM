# Reproducing every table and figure

Table and figure numbers refer to the camera-ready paper (AACL-IJCNLP 2026).
Each result comes from one of three sources:

| Source | Code | Tables and figures |
|---|---|---|
| **Camera-ready run** (27 Sep 2026) | `pipelines/v3_patched/camera_ready_extras.py`, driven by `scripts/run_camera_ready.sh` | Table 1; Table 2 (ChatDoctor, CmedqaRetrieval, FiQA-2018); Table 3 (all rows except query smoothing); Tables 7–10; Figure 1; the results quoted in Sections 5–6 |
| **Validation-split pipeline** (review version) | `pipelines/v2_validation/` | Table 6 |
| **Single-configuration runs** (review period) | `pipelines/v3_patched/main_pipeline1_patched.py` | Table 3 query-smoothing row; Table 4 |
| **Earlier pipeline** (review version, not rerun) | `pipelines/v3_patched/main_pipeline1_patched.py`, `pipelines/v1_original/` | Table 2 rows marked ‡; Table 11; the gate-off numbers in Section 6.3 |
| **Analysis scripts** | `scripts/analysis/` | descriptive statistics in Sections 4.1 and 6.2–6.4 (Section 4 below) |

The outputs of the camera-ready run are in `results/camera_ready/`. Paths
below use `$LEXTM_DATA` for the data root (the directory that holds
`ntcir13_MedWeb_TestCollection/` and `beir/`).

---

## 1. Camera-ready run

```bash
export LEXTM_DATA=/path/to/data
bash scripts/run_camera_ready.sh            # -> results/camera_ready_rerun/
```

The script runs these commands from `pipelines/v3_patched/` (each block is
independent; a failure in one does not stop the others):

```bash
CR="python camera_ready_extras.py"
OUT=../../results/camera_ready_rerun

# MedWeb EN/JA/ZH, BGE-M3 and Qwen3-Embedding-0.6B
$CR medweb --medweb_dir $LEXTM_DATA/ntcir13_MedWeb_TestCollection \
    --encoders bge-m3,qwen3-0.6b --out $OUT

# FiQA-2018 (downloaded from BEIR on first use)
$CR beir --dataset fiqa --data_dir $LEXTM_DATA/beir/fiqa --split test --lang en \
    --download --out $OUT

# ChatDoctorRetrieval (export once with export_chatdoctor_to_beir.py)
python export_chatdoctor_to_beir.py --out_dir $LEXTM_DATA/beir/chatdoctor
$CR beir --dataset chatdoctor --data_dir $LEXTM_DATA/beir/chatdoctor \
    --split test --lang en --out $OUT

# CmedqaRetrieval, full corpus (--n_docs 0), dev split; multi-vector skipped
$CR export_cmedqa --out_dir $LEXTM_DATA/beir/cmedqa --n_docs 0
$CR beir --dataset cmedqa --data_dir $LEXTM_DATA/beir/cmedqa --split dev --lang zh \
    --no_multivector --out $OUT

# Macro file read by the paper; FIG1_EN=1 selects the Figure 1 case
FIG1_EN=1 $CR macros --out $OUT
```

Defaults that the paper relies on (command-line values are recorded under
`config` in each JSON; α, β, λ, Gibbs sweeps and tolerance, RRF k, bootstrap
resamples and chunking are constants in the script):
γ = 0.7, τ = 0.8, K = 20 on MedWeb and 50 elsewhere, γ sweep
{0.5, 0.6, 0.8, 0.9}, τ sweep {0.5} (plus τ = 1.1 for LDA routing), Gibbs seeds
42–46 (42 for every main result), RRF k = 60, 10,000 bootstrap resamples,
chunk size 512, SPLADE++ checkpoint `naver/splade-cocondenser-ensembledistil`.

### Where each number is stored

`M` = `results/camera_ready/medweb_results.json`, `L` = `M["langs"][lang]`
with `lang` in `en`, `ja`, `zh`; `B` = `results/camera_ready/beir_<name>_results.json`.
System names: `bm25`; `m3.sparse`, `m3.multivector`, `m3.all` (BGE-M3 learned
sparse, multi-vector, and the three combined); `splade`; and for each encoder
`<enc>` in `bge-m3`, `qwen3-0.6b`: `<enc>.dense`, `<enc>.rrf`, `<enc>.lextm`
(final configuration), `<enc>.lextm_noadapt` (no entropy adaptation),
`<enc>.lda_route` (τ = 1.1, symmetric prior, no adaptation),
`<enc>.topic_only` (γ = 0), `<enc>.lextm_g<γ>` (γ sweep, no adaptation),
`<enc>.lextm_t0.5` (τ = 0.5, no adaptation).

| Paper item | Field |
|---|---|
| Table 1, MRR and 95% CI | `L["systems"][system]["mrr"]`, `["ci"]` |
| Table 1, Δ rows; Table 7 | `L["paired"][enc][other]` → `delta`, `lo`, `hi`, `p` (paired randomization test), `wins`/`losses`/`ties` |
| Table 3 (except smoothing) | `L["systems"]` entries `bge-m3.lextm`, `bge-m3.lextm_noadapt`, `bge-m3.lextm_t0.5`, `bge-m3.lda_route`, `bge-m3.dense` (γ = 1), `bge-m3.lextm_g*`, `bge-m3.topic_only` (γ = 0) |
| Seed study (Section 6.1) | `L["seed_runs"]["runs"][seed]` for seeds 42–46: `lextm`, `lextm_noadapt`, `lda_route` |
| Table 8 (query types) | `L["breakdown"]["labels"][system][k]` with `k` in `"01"`, `"2"`, `"3+"` (number of labels), and `L["breakdown"]["group"][system][k]` with `k` in `"small (<10)"`, `"mid (10-99)"`, `"large (>=100)"` (relevant tweets); each value is `[mean RR, number of label sets]` |
| Table 9 (label-free control) | `L["label_free"]["systems"]`, `L["label_free"]["paired"]` |
| Figure 1 | `M["langs"]["en"]["figure1_candidates"][1]` (selected with `FIG1_EN=1`) |
| BM25 tie ranges (Section 6.4) | `L["bm25_ties"]` (`best` and `worst` tie resolution) |
| Document and query lengths (Section 6.3) | `L["stats"]`, `B["result"]["stats"]` |
| Cross-check against `LexTMRouter.retrieve()` | `L["crosscheck"]` (identical top-20 lists for 39/39 queries in each language) |
| Table 2 rows ChatDoctor, CmedqaRetrieval, FiQA; Table 10 | `B["result"]["systems"][system]["ndcg10"]`, `["mrr10_q"]` (query-level MRR@10), `["mrr"]` (pair-level MRR@20); paired tests in `B["result"]["paired"]` |

Per-query reciprocal ranks for every MedWeb system are in `L["per_query"]`,
in the order of `L["queries"]`, and as CSV in
`results/camera_ready/per_query/medweb_<lang>.csv`
(`python scripts/export_per_query_csv.py results/camera_ready`).

`results/camera_ready/results_generated.tex` is the macro file the paper's
LaTeX source reads (`\R{mw.en.bge-m3.lextm}` and so on); `macros` rebuilds it
from the JSON files.

---

## 2. Validation-split pipeline (Table 6)

Table 6 is the review version's τ selection on seven validation label sets.

```bash
cd pipelines/v2_validation
for tau in 0.5 0.8; do
  for lang in en ja zh; do
    python main_pipeline_valid.py --dataset_name medweb_${lang} \
        --filepath $LEXTM_DATA/ntcir13_MedWeb_TestCollection/medweb_rag_${lang}_fixed.csv \
        --gamma 0.7 --tau $tau --num_topics 20 --chunk_size 512 \
        --val_fraction 0.2
  done
done
```

The printed validation MRR fills the table. Table 6 comes from the review
version's runs; with the split-stability fix (`docs/FIXES.md` #20) the split is
reproducible but may differ from the one used then, so expect different
values. See `docs/TAU_VALIDATION_PROTOCOL.md` for how the choice of τ is
reported.

---

## 3. Earlier pipeline (review version)

These results were not rerun for the camera-ready version (paper Appendix G);
they predate several fixes in `docs/FIXES.md` and serve only to show where
routing does not help.
Commands are given so the runs can be repeated; expect small differences.

### TREC-COVID and NFCorpus (Table 2, ‡)

150-token chunks, K = 50, 300 Gibbs sweeps, sparsity gate δ = 3, τ = 0.5,
top 100 chunks max-pooled per document, official BEIR nDCG@10:

```bash
cd pipelines/v1_original
python run_trec_covid.py            # Dense (γ = 1.0), Lex-TM (γ = 0.7), BM25
python fast_trec_bm25_hybrid.py     # RRF (k = 60) from the saved run files
python run_nfcorpus.py
python fast_nfcorpus_bm25_hybrid.py
```

Both scripts read the BEIR files from `../data/SOTA/beir/<dataset>`; edit
`data_path` at the top to point elsewhere. Their BM25 baselines tokenize
queries (and, for NFCorpus, documents) by whitespace only, so the BM25 column
is weak; TREC-COVID documents use the BM25 tokenizer of the loader version of
that run, reproduced in `run_trec_covid.py` (`tokenize_for_bm25`).

### MIRACL-zh and Mr. TyDi-ja/th (Table 2, ‡; Section 6.3)

```bash
cd pipelines/v3_patched
for ds in miracl_zh mr_tydi_ja mr_tydi_th; do
  split=test; [ "$ds" = miracl_zh ] && split=dev
  python main_pipeline1_patched.py --dataset_name $ds \
      --filepath     $LEXTM_DATA/$ds/corpus.jsonl \
      --queries_path $LEXTM_DATA/$ds/queries.jsonl \
      --qrels_path   $LEXTM_DATA/$ds/qrels/$split.tsv \
      --gamma 0.7 --tau 0.5 --num_topics 50 --min_query_tokens 5 \
      --query_smoothing 0.0 --no_adaptive_gamma
done
```

Pure Dense: the same command with `--gamma 1.0`. Without the sparsity gate
(Section 6.3: MIRACL-zh 0.4246, Mr. TyDi-ja 0.7643, -th 0.7513): the same
command with `--min_query_tokens 0`. The RRF column (k = 600) comes from the
earlier baseline script `pipelines/v1_original/run_bm25_hybrid_baseline.py`
(see `docs/FIXES.md` #1).

### News and review collections (Table 11)

```bash
cd pipelines/v3_patched
for ds in thucnews sogou english_aviation chinese_aviation; do
  python main_pipeline1_patched.py --dataset_name $ds --filepath <file for $ds> \
      --gamma 0.7 --tau 0.5 --num_topics 50 \
      --query_smoothing 0.0 --no_adaptive_gamma
done
```

Pure Dense: `--gamma 1.0`; without the HHI prior (Appendix H: 0.0896 and
0.4007 on the review collections): `--tau 1.1`. BM25 and RRF (k = 600) on
THUCNews and Sogou come from `pipelines/v1_original/run_bm25_hybrid_baseline.py`;
they were not run on the review collections.

### Query smoothing row of Table 3

An earlier diagnostic run with entropy adaptation and query smoothing 0.1
(condition 4 of `pipelines/v3_patched/diagnose_medweb.sh`):

```bash
cd pipelines/v3_patched
python main_pipeline1_patched.py --dataset_name medweb_en \
    --filepath $LEXTM_DATA/ntcir13_MedWeb_TestCollection/medweb_rag_en_fixed.csv \
    --gamma 0.7 --tau 0.8 --num_topics 20 --chunk_size 512 \
    --query_smoothing 0.1 --adaptive_gamma
```

(`medweb_ja` and `medweb_zh` add `--chunk_size_ja 512` / `--chunk_size_zh 512`.)

### Table 4 (top words of matching topics)

Every `main_pipeline1_patched.py` run writes the top ten words of each topic
to `experiment_logs1/<dataset>_gamma_<γ>_tau_<τ>_topics.json`. Run the MedWeb
command above with `--tau 1.1 --no_adaptive_gamma` (LDA routing) and with
`--tau 0.8`, then pair each LDA-routing topic with the Lex-TM topic that shares
most of its ten top words:

```bash
python scripts/match_topics.py \
    pipelines/v3_patched/experiment_logs1/medweb_en_gamma_0.7_tau_1.1_topics.json \
    pipelines/v3_patched/experiment_logs1/medweb_en_gamma_0.7_tau_0.8_topics.json
```

Table 4 shows the leading top words of the rows chosen in the paper (K = 20,
seed 42). The script also prints the mean number of shared top words over the
best-matched pairs, the overlap quoted in Section 6.2.

---

## 4. Data preparation and descriptive statistics

```bash
# MedWeb retrieval files from the six NTCIR-13 MedWeb files (byte-identical to the
# files used in the paper): medweb_rag_{en,ja,zh}_fixed.csv
python pipelines/v3_patched/data_prep/build_medweb_csv.py --medweb_dir $LEXTM_DATA/ntcir13_MedWeb_TestCollection

# MIRACL-zh and Mr. TyDi-ja/th, 100K-passage samples keeping every judged passage
python pipelines/v3_patched/data_prep/download_miracl_chinese.py --output_dir $LEXTM_DATA/miracl_zh
python pipelines/v3_patched/data_prep/download_mr_tydi.py --language ja --output_dir $LEXTM_DATA/mr_tydi_ja
python pipelines/v3_patched/data_prep/download_mr_tydi.py --language th --output_dir $LEXTM_DATA/mr_tydi_th

# Section 6.2-6.4: HHI vs document frequency, count-one share, amplified terms,
# raw tweet lengths (its BM25 values use rank_bm25 on its own index and are a
# cross-check only; the paper's BM25, tie-range and label-free numbers come from
# the camera-ready JSON)
python scripts/analysis/medweb_prior_stats.py --medweb_dir $LEXTM_DATA/ntcir13_MedWeb_TestCollection
# Section 6.2 and 4.1: query-term document frequencies and E_v, label-set sizes and label counts
python scripts/analysis/medweb_query_terms.py --medweb_dir $LEXTM_DATA/ntcir13_MedWeb_TestCollection
# Section 6.3: document and query lengths of NFCorpus, TREC-COVID and ChatDoctor
python scripts/analysis/beir_length_stats.py --beir_dir $LEXTM_DATA/beir --chatdoctor_dir $LEXTM_DATA/beir/chatdoctor
```

The share of a ChatDoctor question's distinct content words that occur in its
answer (16%, Section 5.2) was computed ad hoc and has no script here.
