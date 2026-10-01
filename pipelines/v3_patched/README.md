# v3_patched — current pipeline

Produces every number in the camera-ready paper except Table 6
(`../v2_validation/`) and the TREC-COVID and NFCorpus rows of Table 2
(`../v1_original/`): `camera_ready_extras.py` for the camera-ready run,
`main_pipeline1_patched.py` for single configurations and the
earlier-pipeline MIRACL, Mr. TyDi, news and review runs. See `../../docs/REPRODUCING_TABLES.md` for the command behind each table
and `../../docs/FIXES.md` for the changelog.

| File | Role |
|---|---|
| `lex_tm_model.py` | HHI-scaled LDA prior and the Numba-compiled collapsed Gibbs sampler (identical in all pipeline versions) |
| `rag_dataload.py` | dataset loaders, per-language tokenization (NLTK, MeCab via fugashi, jieba), chunking, vocabulary |
| `rag_evaluation_pipeline_patched.py` | `LexTMRouter` (routing score, entropy adaptation, sparsity gate), `HybridRouter` (BM25, RRF), evaluation, bootstrap and paired bootstrap tests |
| `main_pipeline1_patched.py` | single-configuration driver; `--run_bm25` adds Pure BM25, RRF (k = 60) and paired tests |
| `camera_ready_extras.py` | every experiment of the camera-ready run (`medweb`, `beir`, `export_cmedqa`, `macros`, `selftest`) |
| `export_chatdoctor_to_beir.py` | writes ChatDoctorRetrieval in BEIR format |
| `export_cmedqa_to_beir.py` | writes CmedqaRetrieval in BEIR format with a seeded, gold-preserving subsample (the paper uses the full corpus via `camera_ready_extras.py export_cmedqa --n_docs 0`) |
| `run_all.sh` | earlier-pipeline suite over all datasets (`--medweb-only`, `--skip-medweb`, `--skip-copy`) |
| `diagnose_medweb.sh` | crosses τ, entropy adaptation and query smoothing on MedWeb; the evidence behind `FIXES.md` #9–#10 and the query-smoothing row of Table 3 |
| `copy_cache.sh` | copies embedding caches named `cache_<dataset>_tau<τ>_K<K>_embeddings.npy` (written by earlier versions) from `experiment_logs1/` to `experiment_logs/` (`v2_validation`); this version names its caches `emb_<dataset>_<model>_chunk<N>.npy` |
| `data_prep/build_medweb_csv.py` | builds `medweb_rag_{en,ja,zh}_fixed.csv` from the six NTCIR-13 MedWeb files |
| `data_prep/download_miracl_chinese.py`, `data_prep/download_mr_tydi.py` | MIRACL-zh and Mr. TyDi-ja/th samples (earlier-pipeline rows of Table 2) |
