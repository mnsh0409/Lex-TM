# v1_original — earlier pipeline, provenance only

**Do not use this to reproduce the camera-ready results.** These files are the
earlier pipeline, kept so that the debugging history in `../../docs/FIXES.md`
can be checked and so that the earlier-pipeline rows of the paper can be
traced to code. Known issues are documented, not fixed.

Earlier-pipeline runs reported in the paper (Table 2, rows marked ‡):

- `run_trec_covid.py`, `run_nfcorpus.py`: Dense (γ = 1.0), Lex-TM (γ = 0.7) and
  BM25 on TREC-COVID and NFCorpus (150-token chunks, K = 50, 300 Gibbs
  sweeps, sparsity gate δ = 3, τ = 0.5, top 100 chunks max-pooled per document,
  official BEIR nDCG@10). They save run files to `result/`. `run_trec_covid.py`
  carries its own `tokenize_for_bm25()`, the BM25 tokenizer of the loader
  version used for that run, which `rag_dataload.py` here lacks.
- `fast_trec_bm25_hybrid.py`, `fast_nfcorpus_bm25_hybrid.py`: RRF (k = 60) and
  a min–max hybrid from those run files.
- `run_bm25_hybrid_baseline.py`: the earlier baseline script whose RRF uses
  k = 600 (MIRACL, Mr. TyDi, THUCNews, Sogou; `FIXES.md` #1).

Known issues (see `docs/FIXES.md`):
- `--run_bm25` is parsed but never wired to `HybridRouter` (#2).
- `run_experiment1.sh`: a malformed default `MAX_DOCS_GENERAL` and a missing
  line continuation make both THUCNews runs fail (#34). The orphaned `done`
  of #12 is not present in this copy.
- RRF uses k = 600 in `run_bm25_hybrid_baseline.py` (#1).
- The BM25 baselines of the TREC-COVID and NFCorpus scripts tokenize queries
  (and, for NFCorpus, documents) by whitespace only.

Use `../v3_patched/` for everything else.
