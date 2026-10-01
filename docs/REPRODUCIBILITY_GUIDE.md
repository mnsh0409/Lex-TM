# Reproducibility guide

Setup, data, pipeline versions and caching for the code of
"Escaping the Tokenizer Trap: Probabilistic Semantic Routing for Dense
Retrieval in Specialized Domains" (AACL-IJCNLP 2026). For the command behind
each table and figure, see `REPRODUCING_TABLES.md`; for every issue found and
fixed, see `FIXES.md`.

## Pipeline versions

| Version | Status | Use for |
|---|---|---|
| `pipelines/v3_patched/` | **current** | every camera-ready number (with `camera_ready_extras.py`) and the earlier-pipeline MIRACL, Mr. TyDi, news and review runs |
| `pipelines/v2_validation/` | superseded | the review version's validation sweep (paper Table 6) |
| `pipelines/v1_original/` | superseded, known bugs documented but not fixed | provenance; the TREC-COVID and NFCorpus scripts and the k = 600 RRF baseline script behind the earlier-pipeline rows |

`lex_tm_model.py` is the same file in all three versions; the topic model was
never changed during debugging (for this release, four comment and message
lines were edited in all three copies):

```bash
md5sum pipelines/*/lex_tm_model.py
```

`rag_dataload.py` in `v1_original/` is the earliest copy. The copies in
`v2_validation/` and `v3_patched/` add the English tokenizer fallback (#15),
and `v3_patched/` also prints the BEIR dataset name in its log (#22).
The fallback is **not** behaviour-preserving (#30): English results computed
without NLTK `punkt` differ from the paper, which is why
`camera_ready_extras.py` refuses to run without it.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords wordnet
```

Python 3.10+. `numpy<2` is required by the pinned `numba` Gibbs sampler. A
CUDA GPU is recommended for the dense encoders (CPU works, 10–30 times
slower); the topic model runs on CPU.

## Data

No dataset is bundled. `README.md` lists where to obtain each one. The shell
scripts in `pipelines/` expect a sibling `../data/` tree (see the path
variables at the top of each script); `scripts/run_camera_ready.sh` reads the
data root from `$LEXTM_DATA`.

## Dense encoders

`BAAI/bge-m3` needs no prompt. Instruction-tuned encoders such as
`Qwen/Qwen3-Embedding-0.6B` need the query-side prompt
(`--query_prompt_name query` in `main_pipeline1_patched.py`; automatic in
`camera_ready_extras.py`); documents are never prompted (#8). Any
`sentence-transformers` model can be passed with `--dense_model`.

## Caching

`main_pipeline1_patched.py` caches embeddings in `experiment_logs1/`, keyed by
(dataset, dense model, chunk size), and topic distributions keyed by
(dataset, τ, K). Gibbs sampling never sees the embeddings, so swapping the
encoder does not recompute topics, and changing τ or K does not re-encode.
`camera_ready_extras.py` reuses these embedding caches after a spot check
(cosine > 0.999 on three random documents) and re-encodes otherwise.

## Determinism

The paper's main results use Gibbs seed 42; Section 6.1 repeats MedWeb with
seeds 43–46 to measure how much the sampler moves MRR. `GIBBS_DETERMINISM_FIX.md`
describes an open issue (`FIXES.md` #38): within one process the sampler seeds
only its first checkpoint chunk, and runs of an identical configuration in separate sessions
have differed by up to 0.01–0.03 MRR. Dense retrieval, BM25 and RRF do not
depend on the sampler and reproduce exactly on the same code version and
device (#23, #24).
