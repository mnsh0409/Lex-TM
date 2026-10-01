# FIXES.md — Complete Changelog

This document itemises every issue discovered while debugging the Lex-TM
pipeline, in the order they were found. Each entry states the symptom, root
cause, fix, and current status. This is provided for transparency: the same
codebase produced several different (and sometimes contradictory) sets of
results over the course of debugging, and readers should be able to
see exactly what changed and why.

**If you only read one thing:** use `pipelines/v3_patched/`. Everything
below explains how we got there. Entries #25–#38 cover the camera-ready run.

Some entries mention scripts used while the paper was under review
(a strong-encoder script, `run_bm25_vs_lextm_medweb.sh`,
`run_medweb_today.sh`, `find_figure1_example.py`). They are not part of
this release; `camera_ready_extras.py` covers what they did (strong
encoder, Pure BM25 and paired tests, MedWeb runs, Figure 1 case mining).

---

### 1. RRF `k` constant inconsistency (k=600 vs k=60)

- **Where:** `run_bm25_hybrid_baseline.py` (v1) vs the validation pipeline's
  `retrieve_rrf()`.
- **Symptom:** Two different Reciprocal Rank Fusion implementations existed
  in the codebase with different `k` constants: one used `k_constant=600`
  (default parameter), the other used `k=60`.
- **Why it matters:** RRF's `k` controls how much rank position is
  discounted. `k=60` is the standard from Cormack et al. (2009) and is what
  the paper reports. `k=600` produces a materially weaker (flatter) RRF
  baseline — an order-of-magnitude difference that would make Lex-TM look
  better than a fair comparison warrants.
- **Fix:** `v3_patched/rag_evaluation_pipeline_patched.py` uses `k=60`
  uniformly, and `main_pipeline1_patched.py` passes `k=60` explicitly at the
  call site rather than relying on a default that could silently drift.
- **Status:** Fixed in v3.

---

### 2. `--run_bm25` flag was dead code (v1)

- **Where:** `pipelines/v1_original/main_pipeline1.py`.
- **Symptom:** The script defines a `--run_bm25` CLI flag via `argparse`,
  but the flag's value is **never read** anywhere in the pipeline body.
  `HybridRouter` (which implements RRF) is defined in
  `rag_evaluation_pipeline.py` but is **never imported** into
  `main_pipeline1.py`.
- **Why it matters:** This means the v1 pipeline, run however you configure
  it, produces **only** Lex-TM / Pure Dense / Standard-LDA numbers — never
  RRF. Any RRF numbers attributed to this pipeline could not have come from
  it; they would have had to come from the separate, differently-configured
  `run_bm25_hybrid_baseline.py` script (see issue #1), or not exist at all.
- **Fix:** `v3_patched/main_pipeline1_patched.py` instantiates `HybridRouter`
  when `--run_bm25` is passed, and calls `evaluate_retrieval_rrf()` +
  `bootstrap_mrr_ci()` on the RRF scores in the same run as the Lex-TM
  evaluation, guaranteeing both numbers come from an identical corpus,
  tokenizer, and query set.
- **Status:** Fixed in v3.

---

### 3. `evaluate_retrieval()` returned `None` instead of per-query scores

- **Where:** `rag_evaluation_pipeline.py` (v1).
- **Symptom:** The function prints MRR/Recall@K but returns nothing,
  discarding the per-query reciprocal-rank scores after computing them.
- **Why it matters:** Any downstream statistical test (e.g. bootstrap CIs)
  cannot reuse these scores and must recompute them independently — which is
  what issue #4 describes.
- **Fix:** `v3_patched` returns `rr_scores: List[float]`, one value per
  query, so callers can pass them directly to `bootstrap_mrr_ci()`.
- **Status:** Fixed in v3.

---

### 4. `bootstrap_mrr_ci()` recomputed retrieval from scratch on every call

- **Where:** `rag_evaluation_pipeline.py` (v1).
- **Symptom:** The original signature was
  `bootstrap_mrr_ci(router, evaluation_dataset, gamma, ...)` — it re-ran
  full retrieval internally rather than accepting pre-computed scores.
  Combined with issue #3 (no scores returned from `evaluate_retrieval`),
  there was no way to compute a bootstrap CI without a second, redundant,
  full pass over every query.
- **Why it matters:** For large evaluation sets this is a significant,
  avoidable cost (retrieval is the expensive step — dense encoding,
  Jensen-Shannon computation, and BM25 scoring for every query, repeated for
  every one of `n_bootstrap` resamples).
- **Fix:** `v3_patched/bootstrap_mrr_ci(rr_scores, n_bootstrap=1000, ...)`
  takes the pre-computed list directly and resamples with `numpy`'s
  `default_rng` (seeded, for reproducibility). No retrieval is re-run.
- **Status:** Fixed in v3.

---

### 5. `HybridRouter.retrieve_rrf()` used an O(N) lookup per result

- **Where:** `rag_evaluation_pipeline.py` (v1), inside `retrieve_rrf()`.
- **Symptom:** `self.corpus_chunk_ids.index(cid)` was called once per
  returned result — a linear scan over the entire corpus for every single
  item in the top-k list, every query.
- **Why it matters:** For a corpus of size N and top_k results, this is
  O(N·top_k) per query — on larger corpora (THUCNews, TREC-COVID) this
  becomes a meaningful bottleneck.
- **Fix:** `v3_patched` builds `self._chunk_id_to_idx: Dict[str, int]` once,
  at `index_corpus()` time, giving O(1) lookup per result.
- **Status:** Fixed in v3.

---

### 6. Cache directory mismatch between pipeline versions

- **Where:** v1 (`run_bm25_hybrid_baseline.py`) used `experiment_logs1/`;
  the intermediate validation pipeline used `experiment_logs/`.
- **Symptom:** Embeddings computed under one pipeline were not automatically
  visible to the other, forcing redundant (expensive) BGE-m3 re-encoding.
- **Fix:** `pipelines/v3_patched/copy_cache.sh` bridges the two directories
  by copying only `_embeddings.npy` files (never `_topicdists.npy`, which are
  hyperparameter-dependent — see #11). The final v3 cache scheme (below)
  supersedes this further.
- **Status:** Mitigated via `copy_cache.sh`; superseded by #11's cleaner
  model-aware keying in the current pipeline.

---

### 7. Tokenizer consistency (verified — not a bug)

- **Question raised:** Do BM25 and Lex-TM use the same tokenizer?
- **Finding:** Yes. Both `LexTMRouter._get_bow()` and
  `HybridRouter.index_corpus()` / `retrieve_rrf()` call
  `self.dictionary.clean_and_tokenize(text, self.language)` — the identical
  method, on the identical `UniversalRAGLoader` instance, with the identical
  language code, at both index time and query time. This was verified by
  direct code inspection, not assumed.
- **Status:** No fix needed. Documented here because it was explicitly
  checked and is load-bearing for the fairness of the RRF comparison.

---

### 8. Query-side prompting for instruction-tuned dense encoders

- **Where:** `v3_patched/rag_evaluation_pipeline_patched.py`,
  `LexTMRouter._encode_query()`.
- **Context:** Added when extending the pipeline to support swapping the
  dense encoder (e.g. `Qwen/Qwen3-Embedding-*`) for the
  strong-retriever ablation.
- **Why it matters:** Instruction-tuned embedding models (Qwen3-Embedding)
  are trained to expect a task instruction prepended to **queries only**
  (documents are encoded without it). Per the model's own documentation,
  omitting this prompt on the query side measurably degrades retrieval
  quality. Silently omitting it would unfairly weaken the strong-retriever
  baseline and invalidate the ablation's conclusion.
- **Fix:** `query_prompt_name` is a constructor parameter (`None` for
  BGE-m3, `"query"` for Qwen3-Embedding), applied only in
  `_encode_query()`; `index_corpus()`'s document encoding path never
  receives a prompt, preserving the intended asymmetric retrieval setup.
- **Status:** Implemented in v3_patched. **Any future encoder
  swap must verify whether it requires a query prompt** before reporting
  numbers — this is not automatic.

---

### 9. MedWeb `tau` selection: 0.5 → 0.8

- **Context:** The Appendix validation sweep (20% held-out validation split,
  never touching the test set) reports validation MRR at `tau ∈ {0.5, 0.8}`
  for each MedWeb language. A later audit of this table found:

  | Language | Val MRR (τ=0.5) | Val MRR (τ=0.8) |
  |---|---|---|
  | English  | 0.7440 | **0.8681** |
  | Chinese  | **0.8681** | 0.8016 |
  | Japanese | **0.7347** | 0.7262 |

  Per-language optimisation would pick different τ per language (0.8 / 0.5 /
  0.5). To avoid overfitting the threshold to individual partitions,
  **aggregate** cross-lingual validation MRR was used as the selection
  criterion instead: τ=0.8 scores 2.396 vs. 2.347 for τ=0.5, so **τ=0.8 was
  adopted uniformly for MedWeb.**
- **Fix applied to this repo:** `pipelines/v3_patched/run_all.sh` (MedWeb
  section only) and `run_medweb_today.sh` (removed in this release, #35)
  were updated from `--tau 0.5` to
  `--tau 0.8` so that running them reproduces the numbers reported in the
  paper. **This change is scoped to MedWeb only** — the other nine datasets
  (aviation, MIRACL, Mr. TyDi, THUCNews, Sogou, TREC-COVID, NFCorpus) were
  never re-validated at τ=0.8 and remain at τ=0.5 in `run_all.sh`.
  `diagnose_medweb.sh` intentionally still sweeps both values — that is its
  purpose and it was left unchanged.
- **Status:** Fixed in v3 run scripts. If you re-run `v1_original` or
  `v2_validation`, you will get the older τ=0.5 MedWeb numbers by design —
  those pipelines are kept for provenance, not for reproducing the paper.
- **Camera-ready note:** the camera-ready paper evaluates all 39 label sets,
  including the seven validation sets, and reports the τ sweep in full;
  see #27 and #28.

---

### 10. `query_smoothing` was found to hurt MRR — disabled by default

- **Context:** An earlier attempted improvement blended each query's topic
  distribution toward the uniform prior (`query_smoothing=0.1`) to stabilise
  noisy short-query inference. Combined with `adaptive_gamma` (which reduces
  the dense weight when the topic signal looks confident), this was the
  active configuration for one submission draft.
- **Diagnostic finding:** A controlled ablation (isolating `tau`,
  `adaptive_gamma`, and `query_smoothing` independently) showed
  `query_smoothing=0.1` **reduced** MRR on **all three** MedWeb languages
  relative to `query_smoothing=0.0`, regardless of `tau` or
  `adaptive_gamma` setting. `adaptive_gamma` alone was neutral-to-mildly
  positive.
- **Fix:** `query_smoothing` defaults to `0.0` (off) in all `v3_patched` run
  scripts. The parameter and code path remain in
  `rag_evaluation_pipeline_patched.py` (`LexTMRouter.query_smoothing`) as an
  ablation option — it is not deleted, only disabled by default — so the
  negative result remains reproducible/inspectable rather than hidden.
  `adaptive_gamma` remains enabled by default.
- **Status:** Fixed (disabled) in v3. This is a documented **negative
  result**: if you are extending this codebase, do not re-enable
  `query_smoothing` without new evidence it helps on your setting.

---

### 11. Model-aware cache keying (introduced with multi-encoder support)

- **Where:** `v3_patched/main_pipeline1_patched.py` cache path construction;
  `rag_evaluation_pipeline_patched.py` `index_corpus()`.
- **Why:** Once the pipeline supported swapping the dense encoder (#8), a
  single shared cache prefix became dangerous: a Qwen3-Embedding run could
  silently load BGE-m3 embeddings left over from a previous run (same
  dataset/tau/K, different model), producing meaningless results with no
  error raised.
- **Fix:** Embedding and topic-distribution caches are now **separate
  files**, keyed differently:
  - `emb_<dataset>_<sanitised_model_name>_chunk<size>.npy` — model-specific.
  - `tdist_<dataset>_tau<τ>_K<K>.npy` — model-**independent** (Gibbs
    sampling on bag-of-words never sees embeddings, so these are safely
    shared and reused across encoders).

  This also means swapping encoders for the strong-encoder ablation does not
  require re-running the Gibbs sampler.
- **Status:** Implemented in v3_patched.
  `index_corpus(chunk_dicts, cache_path=...)` (the old single-prefix API)
  is still accepted for backward compatibility, but new code should pass
  `emb_cache_path` and `tdist_cache_path` separately.

---

### 12. Pre-existing syntax error in `v1_original/run_experiment1.sh`

- **Where:** `pipelines/v1_original/run_experiment1.sh`, line 196.
- **Finding:** `bash -n run_experiment1.sh` fails with
  `syntax error near unexpected token 'done'`. There is exactly one `for`
  loop in the file (opens line 112, correctly closes line 171), and then an
  **orphaned `done`** at line 196 with no matching loop opener.
- **Impact:** Because this is a top-level parse error, `bash
  run_experiment1.sh` cannot execute **any** part of the script — not just
  the code after line 196. This was verified against the original version
  of the script.
- **Fix:** **None applied.** `v1_original/` is kept for provenance; its bugs
  are documented, not fixed. This is documented here rather than
  silently fixed so the historical record is honest. It is independent
  confirmation of why `v1_original` should never be used to reproduce
  results — even setting aside issues #1-#6, the original orchestration
  script did not run end-to-end.
- **Status:** Documented, not fixed (by design). Use `v3_patched/run_all.sh`
  instead, which passes `bash -n` cleanly.
- **Update (camera-ready audit):** the copy in this repository does not contain
  the orphaned `done` and passes `bash -n`; it is a later copy than the one
  described above. It has two other bugs (#34).

---

### 13. `bfloat16` embeddings crashed on `.numpy()` conversion

- **Where:** `rag_evaluation_pipeline_patched.py`, `LexTMRouter._encode_query()`
  and `index_corpus()`'s document-embedding branch.
- **Symptom:** `TypeError: Got unsupported ScalarType BFloat16`, raised from
  inside `np.save(emb_file, self.dense_embeddings.cpu().numpy())` (and would
  equally have hit the `torch.matmul(...).cpu().numpy()` calls in
  `retrieve()`/`retrieve_rrf()` had it gotten that far).
- **Root cause:** Some HuggingFace checkpoints — Qwen3-Embedding among them —
  load in `bfloat16` by default when no explicit dtype override is given
  (`--model_dtype` unset). `numpy` has no native `bfloat16` dtype, so any
  `.numpy()` call on a bf16 `torch.Tensor` raises this `TypeError`
  regardless of what operation triggered it. This only surfaces when
  swapping in an alternate encoder (#8) — BGE-m3 does not load in bf16 by
  default, so this was invisible until the strong-retriever
  ablation exercised a different model.
- **Fix:** Both `_encode_query()` and the document-embedding branch of
  `index_corpus()` now call `.float()` immediately after `.encode(...)`,
  casting to float32 once at the source. Every downstream consumer
  (`np.save`, `torch.matmul`, cosine similarity) then operates on a
  known-safe dtype regardless of what precision the underlying model loads
  in. This is a no-op (and negligible cost) for encoders that already
  return float32, so BGE-m3 runs are unaffected.
- **Blast radius:** No corrupted cache files result from this failure — the
  crash occurs while evaluating `.cpu().numpy()` as an argument to
  `np.save()`, before `np.save()` itself is invoked, so no partial `.npy`
  file is ever written. Any run that hit this error can simply be re-run
  after pulling the fix; no cache cleanup is required.
- **Status:** Fixed in v3_patched (`md5sum`-verified identical).

---

### 14. CPU/GPU device mismatch between cached and freshly-computed embeddings

- **Where:** `rag_evaluation_pipeline_patched.py`, `index_corpus()` embedding
  branches; surfaced in `retrieve()` / `retrieve_rrf()` at the
  `torch.matmul(self.dense_embeddings, query_dense)` line.
- **Symptom:** `RuntimeError: Expected all tensors to be on the same device,
  but found at least two devices, cuda:0 and cpu!`
- **Root cause — and why it was intermittent:** `self.dense_embeddings` was
  left on **different devices depending on whether the cache was hit**:
  - Fresh-compute path: `encode(convert_to_tensor=True)` returns a tensor on
    the **model's device** (GPU when present).
  - Cached path: `torch.tensor(np.load(...))` always returns a **CPU**
    tensor.

  The query embedding (`_encode_query()`) follows the model's device (GPU).
  So on a **first** run — embeddings freshly computed, everything on GPU —
  the matmul succeeds. But once a `.npy` cache exists, a **later** run loads
  those embeddings onto CPU while the query stays on GPU, and the matmul
  fails. In the strong-retriever ablation this showed up mid-sweep: the
  earlier EN/JA steps computed fresh (GPU/GPU, fine), then a later step
  loaded ZH embeddings from the cache the earlier step had written
  (CPU/GPU, crash). This is a classic cache-dependent Heisenbug — the code
  path that fails is determined by prior runs' side effects, not by the
  current command.
- **Fix:** Resolve the model device once
  (`getattr(self.dense_model, "device", ...)`, with a
  `torch.cuda.is_available()` fallback) and `.to(target_device)` the
  embeddings in **both** branches. Behaviour is now identical whether or not
  a cache file exists, and the matmul operands are always co-located. On a
  CPU-only box `target_device` resolves to `cpu` and everything still works.
- **Interaction with #13:** both fixes touch the same two lines; the final
  form is `encode(...).float().to(target_device)` for the fresh path and
  `torch.tensor(np.load(...), dtype=torch.float32).to(target_device)` for
  the cached path. `.float()` fixes dtype (#13), `.to()` fixes device (#14).
- **Status:** Fixed in v3_patched (`md5sum`-verified identical).

---

### 15. NLTK `punkt` resource failure — English tokenizer fallback

- **Where:** `rag_dataload.py`, `clean_and_tokenize()`, English branch
  (`word_tokenize`).
- **Symptom:** `LookupError: Resource punkt not found ... Attempted to load
  tokenizers/punkt/PY3/english.pickle`, causing all English-partition runs
  to fail while Japanese (fugashi/MeCab) and Chinese (jieba) runs — which do
  not use NLTK — completed normally.
- **Root cause:** A persistent version-layout mismatch in NLTK. The loader
  requests the legacy `punkt/PY3/*.pickle` layout, but depending on the
  installed NLTK version and what `nltk.download('punkt')` fetches, the
  on-disk layout may not match, and the resource fails to resolve. This
  proved environment-specific and did not reliably clear via re-downloading
  `punkt`/`punkt_tab` (the standard remedy) in the affected conda env.
- **Fix:** The English branch now wraps `word_tokenize` in a `try/except
  LookupError` and falls back to a regex word-boundary tokenizer
  (`re.findall(r"[A-Za-z]+", text.lower())`) when punkt is unavailable.
  **This is behaviour-preserving for this pipeline:** the very next step
  filters tokens to `.isalpha()` and `len > 2`, so the only tokens that
  survive are alphabetic words — exactly what the regex extracts. Verified
  on representative MedWeb-style English text that the surviving token set is
  identical between the two tokenizers after the existing filter (contraction
  edge cases like "don't" differ pre-filter but all fragments are dropped by
  `.isalpha()`/`len > 2`, leaving identical BoW). English MRR numbers are
  therefore unaffected on machines where punkt *does* load — the fallback
  only changes behaviour when the alternative was a hard crash.
- **Scope:** Applied to the `v2_validation/` and `v3_patched/`
  copies of `rag_dataload.py`. **`v1_original/` is intentionally left
  unpatched** (kept for provenance). This is the first
  fix to modify `rag_dataload.py`; see `docs/REPRODUCIBILITY_GUIDE.md`.
- **Alternative (no code change):** If you prefer to keep NLTK tokenization,
  the resource can sometimes be forced into place with:
  `python -m nltk.downloader -d ~/nltk_data punkt punkt_tab wordnet stopwords`
  run inside the target environment. The code fallback exists because this
  did not reliably resolve in all environments.
- **Status:** Fixed in v2/v3. **Correction:** the fallback is *not*
  behaviour-preserving; see #30.

---

### 16. CVE-2025-32434 blocks Qwen weight loading (torch<2.6 + .bin fallback)

- **Where:** `rag_evaluation_pipeline_patched.py`, `SentenceTransformer(...)`
  construction; surfaced from `find_figure1_example.py` (and any run that
  loads a model whose weights resolve to a legacy `pytorch_model.bin`).
- **Symptom:** `ValueError: Due to a serious vulnerability issue in
  torch.load ... upgrade torch to at least v2.6 ... does not apply when
  loading files with safetensors.`
- **Root cause:** `transformers` >= 4.48 refuses to `torch.load` legacy `.bin`
  weights unless `torch` >= 2.6, because of CVE-2025-32434 (an RCE in
  `torch.load` even with `weights_only=True`). The restriction is waived for
  `.safetensors` weights. The strong-retriever sweep (JA/ZH) loaded Qwen from
  safetensors and worked; the failure appeared when a code path resolved to a
  `.bin` file instead (partial/*mismatched cache, or from_pretrained choosing
  .bin when safetensors was not explicitly requested).
- **Fix:** `LexTMRouter.__init__` now defaults `use_safetensors=True` and
  forwards it through `model_kwargs` to `from_pretrained`, forcing the
  safetensors weights and refusing the `.bin` path. This avoids both a global
  `torch>=2.6` upgrade (which risks disturbing a working CUDA/torch stack) and
  the dangerous workaround of disabling `check_torch_load_is_safe()` in
  transformers (which would silently remove an RCE protection library-wide —
  explicitly NOT done here). Opt out with `use_safetensors=False` only for a
  model that genuinely ships no safetensors weights, in which case torch>=2.6
  is the correct fix.
- **Status:** Fixed in v3_patched.

---

### 17. Added Pure BM25 baseline (lexical-overlap test)

- **Where:** `rag_evaluation_pipeline_patched.py` (`HybridRouter.retrieve_bm25`,
  `evaluate_retrieval_bm25`); `main_pipeline1_patched.py` (runs under
  `--run_bm25`); `run_bm25_vs_lextm_medweb.sh` (a review-period script, not in this release).
- **Why:** To test whether Lex-TM's gains are lexical-overlap
  artifacts. The pipeline built a BM25 index (inside HybridRouter) but only
  ever used it for RRF fusion — there was no way to report Pure BM25 alone.
  NOTE: `LexTMRouter.retrieve(gamma=0.0)` is NOT Pure BM25 — it is Pure
  Topic-Model (the interpolation is dense×γ + topic×(1−γ), with no BM25 term).
  A dedicated BM25 retrieval + evaluation path was therefore required.
- **What:** `retrieve_bm25()` ranks purely by BM25 score using the same
  `clean_and_tokenize` path as Lex-TM's BoW and the RRF index (no tokenizer
  confound). `evaluate_retrieval_bm25()` reports MRR/Recall on the identical
  eval set. Under `--run_bm25`, Pure BM25 now runs alongside RRF at no extra
  cost (both share the BM25 index). Lex-TM vs BM25 is thus apples-to-apples.
- **Status:** Added to v3_patched.

---

### 18. SELF-INFLICTED REGRESSION: TREC-COVID/NFCorpus prefix bug in this patched pipeline

**This one is different from the others below: it was not found in the
original codebase. It was introduced by an earlier version of this
patched pipeline, and is documented
here in full because a reproducibility package that hides its own mistakes
isn't one.**

- **Where:** `main_pipeline1_patched.py`, `loaders` dict.
- **What the original code actually does:** `v1_original/main_pipeline1.py`
  contains, in a single dict literal:
  ```python
  'trec_covid': loader.load_beir,                                   # bare — defaults to prefix="nfcorpus_"
  'nfcorpus':   loader.load_beir,                                   # bare — defaults to prefix="nfcorpus_" (harmless here)
  ...
  'trec_covid': lambda fp: loader.load_beir(fp, prefix='trec_covid_'),  # duplicate key — CORRECT
  'nfcorpus':   lambda fp: loader.load_beir(fp, prefix='nfcorpus_'),    # duplicate key — CORRECT (redundant)
  ```
  Python dict literals silently keep only the **last** definition of a
  duplicate key. So despite looking redundant/messy, the original code
  **runs correctly** — the corrected lambdas win.
- **What went wrong:** An earlier version of this file (when
  `main_pipeline1_patched.py` was first built from the original) captured only the *first* (bare, buggy) definition of
  `'trec_covid'` and `'nfcorpus'` and dropped the correcting duplicate —
  because a duplicate dict key silently overriding an earlier one is easy
  to miss on a read-through, especially across a large dict.
- **Impact if left unfixed:** `loader.load_beir(fp)` called with no prefix
  argument defaults to `prefix="nfcorpus_"` for BOTH datasets. For
  `nfcorpus` this is harmless (matches the intended prefix by coincidence).
  For `trec_covid`, every document would be tagged `nfcorpus_<id>` instead
  of `trec_covid_<id>`. `_QRELS_DATASETS['trec_covid'] = 'trec_covid_'`
  means `build_evaluation_dataset_qrels()` would then look for
  `trec_covid_<id>` in a `chunk_lookup` that only contains `nfcorpus_<id>`
  keys — every single qrel silently fails the `if doc_id not in
  chunk_lookup` check and gets skipped. Net effect: **zero evaluation
  pairs for TREC-COVID**, with no exception raised — the pipeline would
  print `Generated 0 evaluation pairs` and either crash on `np.mean([])`
  or (depending on downstream handling) produce a meaningless result.
- **Did this corrupt anything already reported?** No. The paper's
  TREC-COVID/NFCorpus numbers (now Table 2) predate this patched
  pipeline and were never regenerated through it. This
  bug was caught by manually cross-checking the loaders dict against the
  original file, before it produced a bad number — not by discovering a
  bad number after the fact. **If TREC-COVID or NFCorpus were EVER
  re-run through `v3_patched` before this fix was applied,
  discard those results and re-run.**
- **Fix:** Rewrote the `loaders` dict entry with explicit prefixes and
  *no duplicate keys* this time — `lambda fp: loader.load_beir(fp,
  prefix='trec_covid_')` and `lambda fp: loader.load_beir(fp,
  prefix='nfcorpus_')` — matching the original's actual (if confusingly
  written) runtime behaviour. Verified via `ast`-based duplicate-key
  scan that no other dict literal in the file has this pattern.
- **Status:** Fixed in v3_patched.

---

### 19. CPU/GPU device mismatch in v2_validation (same root cause as #14, different file)

- **Where:** `pipelines/v2_validation/rag_evaluation_pipeline_valid.py` —
  the embedding load/compute site, plus **two** independent matmul
  consumption sites (`retrieve()` and `retrieve_rrf()`, which do their own
  separate encode+matmul here rather than `retrieve_rrf()` delegating to
  `retrieve()` as it does in v3_patched).
- **Why:** Identical mechanism to #14 — `torch.tensor(np.load(...))`
  defaults to CPU, `SentenceTransformer.encode()` defaults to the model's
  device (GPU if available). First run of a given (dataset, τ, K) computes
  fresh and matches; any subsequent run of the same config loads the
  now-cached CPU tensor and crashes against the GPU-encoded query
  (`RuntimeError: mat is on cpu, different from other tensors on cuda:0`).
  v2_validation was not covered when #14 was originally fixed in
  v3_patched, since the two files diverged independently.
- **Fix:** Same `.to(self.dense_model.device)` / `.to(self.dense_embeddings.device)`
  pattern as #14, applied at all three sites in this file.
- **Status:** Fixed.

---

### 20. Non-deterministic val/test split despite a fixed seed (v2_validation)

- **Where:** `pipelines/v2_validation/main_pipeline_valid.py`, the
  `val_fraction` split logic.
- **What the code looked like:**
  ```python
  unique_queries = list({item['query'] for item in eval_dataset})
  rng = np.random.default_rng(42)
  rng.shuffle(unique_queries)
  ```
- **Why this doesn't do what it looks like it does:** the seed is real, but
  `{item['query'] for item in eval_dataset}` is a Python `set`, and CPython
  randomizes string hashing per-process by default (`PYTHONHASHSEED`,
  since 3.3+). `rng.shuffle()` is deterministic *given its input order* —
  but the input order here was never stable, so every separate `python`
  invocation (i.e. every line of `run_experiment_valid.sh`) drew a
  different effective val/test partition despite the "fixed" seed. The
  seed protected the shuffle algorithm; nothing protected the thing being
  shuffled.
- **How this was caught:** the identical (γ=0.7, τ=0.5) configuration was
  run twice in one script execution — once computing fresh, once reloading
  cached embeddings and cached (deterministically-retrained, identical
  log-likelihood) topic distributions. With everything else held fixed,
  validation MRR still swung from 0.62→0.93 (Chinese) and 0.82→1.00
  (Japanese, on n=7) between the two runs. That swing is only explicable by
  the val/test membership itself changing.
- **Downstream consequence:** any per-language "best τ" comparison made
  using this script's validation numbers is comparing different random
  query subsets across τ values, not the same held-out set — the
  comparison is not measuring what it appears to measure. Re-running the
  three τ sweeps on a different day reproduced a different per-language
  ranking than an earlier run (two of three languages flipped which τ
  scored best), consistent with this being noise rather than signal at
  n=7/n=32 scale.
- **Fix:** `sorted({item['query'] for item in eval_dataset})` — sorting
  fixes the shuffle's input order, making the seeded shuffle actually
  reproducible across runs and machines, as originally intended.
- **Status:** Fixed. **Any numbers previously generated via the unpatched
  version of this script should not be treated as a stable, reproducible
  validation-based hyperparameter selection** — re-run after this fix if
  those numbers are load-bearing for a claim.

---

### 19b. Added paired bootstrap significance test (Lex-TM vs. RRF / BM25)

- **Where:** `rag_evaluation_pipeline_patched.py` (`paired_bootstrap_diff`);
  called automatically from `main_pipeline1_patched.py` whenever
  `--run_bm25` is passed.
- **Why:** Comparing two independently-computed CIs (Lex-TM's CI vs RRF's
  CI, each from `bootstrap_mrr_ci`) is conservative and low-power.
  Overlapping CIs mean "this comparison can't distinguish them" — not
  "parity" — and non-overlapping CIs are suggestive but not a formal test.
  Since Lex-TM and RRF are evaluated on the identical query set, a paired
  test (same resampled query indices applied to both arms per draw,
  difference computed per draw) is the statistically correct comparison
  and has more power to detect a real effect where one exists.
- **What:** `paired_bootstrap_diff(rr_a, rr_b, ...)` resamples query
  indices once per bootstrap draw and applies the SAME indices to both
  score arrays, computing `mean(A[idx]) - mean(B[idx])` per draw. Reports
  the 95% CI on the difference; significant only if that interval excludes
  zero. Self-tested on synthetic data matching both the ZH regime (Lex-TM
  genuinely behind RRF) and the EN regime (Lex-TM genuinely ahead): correctly
  reports "not significant" in the former and "significant" in the latter.
  This function cannot manufacture a win — if the underlying per-query
  scores don't support one, the reported CI straddles zero and the printed
  verdict says so explicitly.
- **Status:** Added to v3_patched (syntax
  verified).

---

### 21. Added CmedqaRetrieval (second Chinese medical benchmark) with disclosed stratified subsampling

- **Where:** `export_cmedqa_to_beir.py`; wired into `main_pipeline1_patched.py`
  (`_DATASET_LANG`, `_QRELS_DATASETS`, loaders dict, argparse choices).
- **Why:** To evaluate on medical benchmarks beyond MedWeb.
  CmedqaRetrieval (Chinese medical Q&A, real patient/physician pairs) is a
  natural second Chinese benchmark.
- **What / honesty note:** The official corpus is 100,001 documents, which is
  not tractable to encode in full for a quick evaluation. The export writes a
  STRATIFIED SUBSAMPLE (default 10,000 docs) that keeps every gold-referenced
  document and fills the rest randomly (seed=42), retaining all 3,999 queries.
  This is NOT comparable to the official full-corpus leaderboard and the export
  prints explicit disclosure text to that effect. The RELATIVE comparison
  (Lex-TM / BM25 / Dense / RRF) remains valid since all methods see the same
  subsample. Result (tau=0.8, disclosed): Lex-TM significantly beats Pure BM25
  (+0.080 MRR), borderline vs RRF (+0.005, CI just includes 0) — consistent
  with the MedWeb Chinese parity boundary.
- **Caveat inherited from source:** CmedqaRetrieval qrels are built from labeled
  Q&A pairs without exhaustive relevance annotation, so some true positives may
  be scored as false negatives. This affects the official benchmark too.
- **Status:** Added to v3_patched.

---

### 22. Misleading hardcoded "NFCorpus" label in the generic BEIR loader

- **Where:** `rag_dataload.py`, `load_beir()` (line ~583).
- **Symptom:** `load_beir()` is generic across all BEIR-format datasets, but
  printed a hardcoded "Loading NFCorpus from: ..." regardless of which dataset
  was actually loading — so running cmedqa printed "Loading NFCorpus" above
  Chinese data, which looks like a data-loading error to anyone reading the log.
- **Root cause:** Stale label left from when the function was first written for
  NFCorpus specifically.
- **Fix:** Derive a readable dataset label from the corpus path's parent
  directory name and print that instead. Cosmetic only — no effect on any
  computation or result, but removes a misleading log line from the
  reproducibility artifacts.
- **Status:** Fixed in v3_patched.

---

### 23. Primary MedWeb table's "Pure Dense" column held Qwen3 numbers, not BGE-m3

- **Where:** `README.md` primary results table; propagated into
  `docs/REPRODUCING_TABLES.md`.
- **Symptom:** The BGE-m3 table's `Pure Dense` column read
  **0.7393 / 0.6323→(shown as 0.7876) / 0.6917** — byte-identical to the
  `Pure Dense (Qwen)` column in the strong-encoder table. Two different
  encoders cannot produce the same MRR to four decimals in three languages.
- **Root cause:** `run_bm25_vs_lextm_medweb.sh` runs a **single arm per
  language** (`--gamma 0.7 --run_bm25`), which yields Lex-TM, Pure BM25 and
  RRF — but **no `--gamma 1.0` arm, therefore no BGE pure-dense number**.
  When the README table was assembled, the only dense logs in existence were
  the Qwen ones (from the strong-encoder script used during review, which *does* run a
  gamma=1.0 arm), and those values were used to fill the column.
- **Fix:** Measured the missing arm directly —
  `--gamma 1.0 --tau 0.8 --num_topics 20 --chunk_size 512 --no_adaptive_gamma`
  with the default encoder (`BAAI/bge-m3`, no query prompt). Result:
  **EN 0.6596, JA 0.6323, ZH 0.6266.** These reproduce the manuscript's dense
  column *exactly* (the Pure Dense row of the paper's Table 1), because at
  gamma=1.0 the topic distribution is ignored in scoring — the Gibbs draw
  never touches the result. This makes the BGE-m3 γ=1.0 arm reproduce the
  manuscript's dense column bit-exact — but note this determinism holds
  *within* a code version and device, not across them: the Qwen γ=1.0 arm
  reproduced exactly on JA and shifted 0.0060 on ZH between two pipeline
  versions (root cause located — see **#24**, resolved). Treat "γ=1.0 is
  deterministic" as version-and-device-scoped, not as a guarantee.
- **Scope of impact:** README and REPRODUCING_TABLES only. No other
  reported number used the BGE pure-dense value: the earlier figures cite Lex-TM vs RRF (+0.11) and
  vs BM25 (+0.17), and the Qwen figures (JA 0.7876→0.8113, ZH 0.6917→0.8248,
  both log-confirmed); none uses BGE pure dense. The correction also
  *widens* the reported Lex-TM margin over dense (EN +0.154 rather than the
  +0.075 the wrong number implied).
- **Status:** Fixed in README and REPRODUCING_TABLES.

---

### 24. Qwen pure-dense (γ=1.0) differs across CODE VERSIONS — resolved, not nondeterminism

- **Where:** `experiment_logs1/medweb_*_qwen3-0.6B_dense.log`, γ=1.0 arm.
- **Symptom:** the "same" command gave two different ZH values across sessions:

  | Lang | run under `main_pipeline1_patchedV3.py` | run under current pipeline | Δ |
  |---|---|---|---|
  | JA | 0.7876 | 0.7876 | 0.0000 — bit-exact |
  | ZH | 0.6917 | 0.6857 | 0.0060 |

- **RESOLVED — it is deterministic, per code version.** Rerunning ZH on the
  current code returns 0.6857 every time. The two values come from two
  different versions of the pipeline, not from run-to-run randomness.
- **Root cause (located):** the device/dtype fix in
  `rag_evaluation_pipeline_patched.py` (~lines 159–185). The cached-embedding
  path loads via `torch.tensor(np.load(...))`, which defaults to CPU and to
  numpy's dtype, while the query embedding follows the *model's* device. The
  fix resolves the model device once and moves embeddings onto it in **both**
  the cached and fresh-encode branches, adding an explicit `.float()` cast.
  Pre-fix and post-fix therefore perform the dense matmul at different
  device/precision, producing last-bit differences in the scores.
- **Why only ZH moved:** at n=39 a single query shifting rank 2→3 changes MRR
  by ~0.0043 and 3→4 by ~0.0021, so ~1–2 near-tied documents swapping order
  fully accounts for 0.0060. Japanese had no ties close enough to flip; hence
  bit-exact. English/Chinese ties are tighter because their score
  distributions are denser near the top.
- **Hypotheses tested and REJECTED along the way (recorded so they are not
  re-litigated):**
  - *NaN/degenerate topic rows poisoning `0.0 * NaN = NaN` at γ=1.0.* Checked
    directly: `tdist_medweb_{en,ja,zh}_tau0.8_K20.npy` contain **0 NaN and 0
    all-zero rows** in all three languages. Not the cause.
  - *Run-to-run float nondeterminism.* Disproved: ZH reproduces exactly on
    repeat runs of the current code.
- **Which value is canonical:** **0.6857** — produced by the current, fixed
  code. 0.6917 is superseded; it was produced before the device/dtype fix.
- **Consequence for #23's claim:** "γ=1.0 is deterministic" is true *within* a
  code version and a device, and is what let BGE-m3 reproduce the manuscript's
  dense column bit-exact. It is **not** a guarantee across versions or devices.
- **Impact (at the time):** the earlier figures quoted
  2 d.p. (0.74→0.86, 0.79→0.81, 0.69→0.82), and the values of that time
  rounded to the same 2 d.p. README Δ updated to the fresh set (EN +0.13, ZH +0.14).
- **Status:** RESOLVED. Canonical Qwen dense at the time = EN 0.7372, JA 0.7876,
  ZH 0.6857 (single sitting, all logged). **Superseded by #31**: the
  camera-ready run gives 0.7436 / 0.7949 / 0.6849, and its Lex-TM values
  (0.8515 / 0.8175 / 0.8241) no longer all round to the earlier 2 d.p. figures.
- **Outstanding caution:** any number in this repo produced *before* the
  device/dtype fix may sit ~0.006 away from what the current code returns.
  The BGE-m3 primary table and the Lex-TM(Qwen) column were not re-measured
  under the current code; they are Gibbs-dependent anyway (±0.01–0.03), which
  dominates this effect.

---

## Camera-ready run (September 2026)

Entries #25–#37 were found while preparing the camera-ready version. The
camera-ready run (`pipelines/v3_patched/camera_ready_extras.py`, outputs in
`results/camera_ready/`) re-measured every MedWeb number in one sitting, with
every paired test computed on identical query lists, and cross-checked its
scoring against `LexTMRouter.retrieve()` (identical top-20 lists for 39/39
queries in each language). Where its numbers differ from earlier ones, the
camera-ready numbers are canonical and are the ones in the paper.

---

### 25. MedWeb headline numbers of the July release were not reproduced

- **Where:** `README.md` and `docs/REPRODUCING_TABLES.md` of the July 2026
  release ("canonical" MedWeb table).
- **Symptom:** the July table gave RRF 0.7209 / 0.6778 / 0.7609 (EN/JA/ZH)
  and Lex-TM 0.8139 in English. The camera-ready run gives RRF
  0.7081 / 0.6883 / 0.7716 and Lex-TM 0.8151, with BM25 and dense retrieval
  unchanged.
- **Root cause:** the camera-ready RRF is the pipeline's own
  `HybridRouter.retrieve_rrf` with k = 60, and its values are identical to the
  pipeline's logged runs in `experiment_logs2a`; the run behind the July RRF
  values was not identified. The July Lex-TM value came from a separate
  sitting under an earlier code version and is 0.0012 below the camera-ready
  value.
- **Effect:** relative gain over RRF in English +15.1% instead of +12.9%. The
  July "held-out title-group evaluation split" label was also wrong: those
  numbers, like the camera-ready ones, cover all 39 label sets.
- **Status:** README and REPRODUCING_TABLES replaced by the camera-ready
  numbers.

---

### 26. Paired differences mixed two configurations

- **Where:** paired tests computed before the camera-ready run.
- **Symptom:** an English difference of +0.11 [+0.05, +0.18] against RRF
  (and +0.17 against BM25) did not match any single run.
- **Root cause:** Lex-TM at τ = 0.5 on the 32 test label sets of the review
  version was paired with baselines evaluated on all 39 label sets.
- **Fix:** `camera_ready_extras.py` computes every paired test from per-query
  arrays of the same run and the same query list (`paired()`), and the paper
  reports +0.107 [+0.041, +0.181] against RRF and +0.159 [+0.055, +0.270]
  against BM25 (Table 1), with a paired randomization test (Table 7).
- **Status:** Fixed.

---

### 27. The submitted English headline used τ = 0.5

- **Where:** the review version of the paper (Table 1).
- **Symptom:** English MRR 0.8312 against RRF 0.7276 on the 32 test label
  sets, although the review version's own validation rule chose τ = 0.8
  (validation MRR summed over the three partitions: 2.396 for τ = 0.8 against
  2.347 for τ = 0.5; paper Table 6).
- **Root cause:** the English cell was taken from the τ = 0.5 ablation row;
  with τ = 0.8 the English test-set MRR was 0.8024. The Japanese cell (0.6849)
  matched no ablation row (see the July `REPRODUCING_TABLES.md`).
- **Fix:** the camera-ready paper uses τ = 0.8 for every partition, reports all
  39 label sets, and states this history in Appendix B.
- **Status:** Fixed in the paper.

---

### 28. How γ and τ were chosen was described too strongly

- **Where:** July `README.md`, `docs/FIXES.md` #9, run scripts.
- **Symptom:** γ = 0.7 was described as selected on the validation split.
- **Root cause:** γ = 0.7 was fixed after early sweeps over γ ∈ {0.7, 0.9, 1.0}
  on all label sets. τ = 0.8 was chosen on seven validation label sets, which
  the reported results include.
- **Effect:** the camera-ready γ sweep shows lower γ does better on MedWeb
  (English 0.8260 at γ = 0.6; Chinese 0.8000 at γ = 0.5; neither difference is
  significant). The paper keeps γ = 0.7 because it was fixed before the sweep,
  and reports both sweeps in full (Table 3). Five Gibbs seeds (42–46) move
  Lex-TM's MRR more than τ does in every language.
- **Status:** Corrected in the paper and in `docs/TAU_VALIDATION_PROTOCOL.md`.

---

### 29. τ sensitivity in Japanese was understated

- **Symptom:** Japanese MRR was described as varying by under one percentage
  point across τ.
- **Fact (camera-ready run, all 39 label sets, seed 42):** τ changes MRR by at
  most 0.0014 in English and Chinese and by 0.0124 in Japanese (paper
  Table 3).
- **Status:** Corrected in the paper.

---

### 30. Correction to #15: the fallback tokenizer is not behaviour-preserving

- **Where:** `rag_dataload.py`, English branch of `clean_and_tokenize()`.
- **Symptom:** a run on a machine where NLTK `punkt` did not load produced an
  English MedWeb vocabulary of 921 terms instead of 909, and Pure BM25 MRR
  0.6327 instead of 0.6566; every English number moves.
- **Root cause:** #15 stated that the regex fallback yields the same tokens
  after the `.isalpha()` and length filters. It does not: `word_tokenize`
  keeps hyphenated words whole and splits contractions as `do` + `n't`, both
  of which the filters then drop, while the regex splits them into parts that
  survive (`runny`, `nose`; `don`).
- **Fix:** `camera_ready_extras.py` stops with an explicit message when `punkt`
  cannot be loaded (`require_punkt()`) and warns when the English MedWeb
  vocabulary is not 909. The fallback in `rag_dataload.py` is kept so that the
  module still imports, but English results computed with it are not
  comparable to the paper.
- **Status:** Fixed (fail fast); #15's "behaviour-preserving" claim withdrawn.

---

### 31. Qwen3-Embedding numbers re-measured; the Japanese gain is not significant

- **Where:** July `README.md` strong-encoder table.
- **Symptom:** the July table gave Qwen3 dense 0.7372 / 0.7876 / 0.6857 and
  Lex-TM 0.8629 / 0.8113 / 0.8248 (EN/JA/ZH) and described the gain as
  "consistently observed in all three languages", without a significance test.
- **Camera-ready run:** dense 0.7436 / 0.7949 / 0.6849, Lex-TM
  0.8515 / 0.8175 / 0.8241. Paired differences over dense retrieval:
  +0.108 [+0.045, +0.178] in English, +0.023 [−0.021, +0.069] in Japanese
  (p = 0.35), +0.139 [+0.064, +0.229] in Chinese.
- **Status:** the paper states a significant gain in English and Chinese only.

---

### 32. Pure BM25 depends on how ties are broken (Japanese, Chinese)

- **Symptom:** Pure BM25 gave 0.5976 (JA) and 0.7708 (ZH) in earlier logs and
  0.5855 and 0.7579 in the camera-ready run.
- **Root cause:** many label strings receive identical BM25 scores, and the
  order in which equal scores are returned differs between implementations
  and runs. Japanese is the most sensitive (24 of the 39 Japanese label
  strings share their bag of words with another label string).
- **Fix:** the camera-ready run records the best and worst tie resolution
  (`bm25_ties`); the earlier values lie inside those ranges (Japanese
  0.5447–0.6347). `FastBM25` gives scores identical to
  `rank_bm25.BM25Okapi`.
- **Status:** Documented; the paper reports the tie range.

---

### 33. CmedqaRetrieval and ChatDoctor rerun under the BEIR protocol

- **Where:** July `README.md` and `REPRODUCING_TABLES.md`.
- **Symptom:** CmedqaRetrieval was evaluated on a 10,000-document stratified
  subsample with pair-level MRR, where Lex-TM beat BM25 and sat at the edge of
  significance against RRF; ChatDoctor was reported as MRR.
- **Camera-ready run:** full CmedqaRetrieval corpus (dev split) and standard
  nDCG@10 for both collections. CmedqaRetrieval: Lex-TM 0.3343 against dense
  0.3372 (it matches dense retrieval, and is above BM25 0.1568 and RRF
  0.2595).
  ChatDoctor: Lex-TM 0.4933 against dense 0.5806.
- **Status:** the subsample numbers are superseded; `export_cmedqa_to_beir.py`
  keeps the subsample option, and `camera_ready_extras.py export_cmedqa
  --n_docs 0` writes the full corpus.

---

### 34. Two further bugs in `v1_original/run_experiment1.sh`

- **Line 64:** `MAX_DOCS_GENERAL="${MAX_DOCS_GENERAL:-0}"#100000` assigns the
  string `0#100000`, not `0` (the `#` does not start a comment inside a word).
- **Line 408:** the THUCNews Lex-TM command lacks a trailing `\` before
  `--max_docs`, so `--max_docs` runs as a separate command.
- **Effect:** both THUCNews runs of the script fail (`--max_docs 0#100000` is
  not an integer). The earlier-pipeline THUCNews numbers were not produced by
  this script (`docs/REPRODUCING_TABLES.md`, Section 3).
- **Also:** the orphaned `done` described in #12 is not present in the copy of
  `run_experiment1.sh` in this repository, which passes `bash -n`; this copy
  is therefore a later copy than the one #12 describes.
- **Status:** Documented, not fixed (provenance, as #12).

---

### 35. `run_medweb_today.sh` summary never showed RRF

- **Symptom:** the RRF column of the script's summary table was always `-`.
- **Root cause:** it grepped `^MRR:` lines and then searched those lines for
  "RRF", which never matched.
- **Status:** the script is removed from the public release; `run_all.sh
  --medweb-only` and `camera_ready_extras.py medweb` cover the same runs.

---

### 36. Per-query values of BEIR collections and the label-free control were not saved

- **Where:** `camera_ready_extras.py`, `cmd_beir()` and the label-free block
  of `cmd_medweb()`.
- **Symptom:** the camera-ready run saved per-query reciprocal ranks for every
  MedWeb system, but only aggregate metrics and paired statistics for
  FiQA-2018, ChatDoctor, CmedqaRetrieval and the label-free control.
- **Fix:** the released script writes them to `per_query/beir_<name>.json`
  (pair-level reciprocal ranks and per-query nDCG@10 for every system) and
  `per_query/medweb_<lang>_label_free.json`. No metric changes.
- **Status:** Fixed for future runs; the released camera-ready outputs keep
  the aggregates for these collections (`results/camera_ready/README.md`).

---

### 37. Claims about Figure 1 and the prior that the camera-ready run did not support

- **Figure 1.** The planned replacement for the original Figure 1 was a
  free-text query without a disease name. MedWeb's protocol has only
  label-string queries, so no such case exists in the evaluation. Figure 1 now
  shows a real case mined from the English run (`figure1_candidates[1]` in
  `results/camera_ready/medweb_results.json`), whose query is a label string.
- **The HHI prior.** The prior was described as emphasising rare, exclusive
  domain terms. On MedWeb's short documents HHI reduces to 1/df, only terms
  with document frequency 2 are amplified at τ = 0.8, and averaged over five
  Gibbs seeds a symmetric prior performs as well (paper Sections 3.1, 6.1
  and 6.2).
- **Status:** Stated in the paper.

---

### 38. Gibbs sampler seeds only its first checkpoint chunk (open)

- **Where:** `lex_tm_model.py`, `LexTMLdaModel.fit()`: the sampler runs in
  chunks of 50 sweeps; the first chunk reseeds Numba's RNG with
  `random_state`, later chunks pass `-1` and continue from the carried RNG
  state.
- **Symptom:** in the review period, runs of an identical configuration in
  separate sessions differed by about 0.01–0.03 MRR.
- **Status:** Open; not fixed in this release, so that the code is the code
  of the camera-ready run. `docs/GIBBS_DETERMINISM_FIX.md` specifies the fix
  and its verification. The paper measures sampler variance with five seeds
  (Section 6.1); bit-exact repetition of the camera-ready run has not been
  verified.

---

## Summary table

| # | Issue | File(s) | Status |
|---|---|---|---|
| 1 | RRF k=600 vs k=60 | `rag_evaluation_pipeline*.py` | Fixed (k=60) |
| 2 | `--run_bm25` dead code | `main_pipeline1.py` | Fixed (wired to `HybridRouter`) |
| 3 | `evaluate_retrieval` returns `None` | `rag_evaluation_pipeline.py` | Fixed (returns `rr_scores`) |
| 4 | Bootstrap re-runs retrieval | `rag_evaluation_pipeline.py` | Fixed (takes pre-computed scores) |
| 5 | O(N) chunk lookup in RRF | `rag_evaluation_pipeline.py` | Fixed (O(1) dict) |
| 6 | Cache dir mismatch | pipeline-wide | Mitigated (`copy_cache.sh`), superseded by #11 |
| 7 | Tokenizer consistency | — | Verified correct, no fix needed |
| 8 | Query-side instruction prompting | `rag_evaluation_pipeline_patched.py` | Implemented |
| 9 | MedWeb tau 0.5→0.8 | `run_all.sh`, `run_medweb_today.sh` (removed, #35) | Fixed (scoped to MedWeb only) |
| 10 | `query_smoothing` hurts MRR | `run_all.sh`, `run_medweb_today.sh` (removed, #35) | Fixed (disabled in scripts; `main_pipeline1_patched.py` defaults to 0.1, pass `--query_smoothing 0.0`) |
| 11 | Cache keyed by model | `main_pipeline1_patched.py` | Implemented |
| 12 | Syntax error in v1 script | `v1_original/run_experiment1.sh` | Documented, not fixed (absent from this copy; see #12 update, #34) |
| 13 | bfloat16 → numpy TypeError | `rag_evaluation_pipeline_patched.py` | Fixed (`.float()` cast at encode time) |
| 14 | CPU/GPU device mismatch (cache-dependent) | `rag_evaluation_pipeline_patched.py` | Fixed (`.to(model_device)` in both paths) |
| 15 | NLTK punkt resource failure (EN only) | `rag_dataload.py` | Fixed (regex fallback); not behaviour-preserving, see #30 |
| 15b | NLTK punkt/PY3_tab OSError + unextracted .zip crashing import | `rag_dataload.py` | Fixed (auto-extract zips; catch OSError; import never crashes) |
| 16 | CVE-2025-32434 blocks Qwen .bin load (torch<2.6) | `rag_evaluation_pipeline_patched.py` | Fixed (force use_safetensors=True) |
| 17 | No Pure BM25 baseline (only RRF used BM25) | `rag_evaluation_pipeline_patched.py` | Added (retrieve_bm25 + evaluator) |
| 18 | **Self-inflicted**: TREC-COVID prefix bug from a dropped duplicate dict key | `main_pipeline1_patched.py` | Fixed — see full writeup, not inherited from original |
| 19 | CPU/GPU device mismatch (same as #14, different file) | `v2_validation/rag_evaluation_pipeline_valid.py` | Fixed — 3 sites |
| 19b | Paired bootstrap significance test added | `rag_evaluation_pipeline_patched.py`, `main_pipeline1_patched.py` | Added |
| 20 | Non-deterministic val/test split despite fixed seed (`set()` order instability) | `v2_validation/main_pipeline_valid.py` | Fixed — see full writeup, invalidates prior split-based hyperparameter comparisons |
| 21 | CmedqaRetrieval added with a disclosed stratified subsample | `export_cmedqa_to_beir.py`, `main_pipeline1_patched.py` | Added; superseded by the full corpus (#33) |
| 22 | Hardcoded "NFCorpus" label in the BEIR loader | `rag_dataload.py` | Fixed (log message only) |
| 23 | MedWeb "Pure Dense" column held Qwen3 numbers | `README.md`, `docs/REPRODUCING_TABLES.md` | Fixed |
| 24 | Qwen pure dense differs across code versions | `rag_evaluation_pipeline_patched.py` | Resolved (deterministic per code version) |
| 25 | July README MedWeb numbers not reproduced | `README.md`, `docs/REPRODUCING_TABLES.md` | Replaced by camera-ready numbers |
| 26 | Paired differences mixed two configurations | paired tests | Fixed (same-run, same-query pairs) |
| 27 | Submitted English headline used τ = 0.5 | review-version paper | Fixed in the paper (Appendix B) |
| 28 | γ, τ selection described too strongly | docs, paper | Corrected |
| 29 | Japanese τ sensitivity understated | paper | Corrected (0.0124) |
| 30 | Fallback tokenizer not behaviour-preserving (corrects #15) | `rag_dataload.py`, `camera_ready_extras.py` | Fixed (fail fast without punkt) |
| 31 | Qwen3 numbers re-measured; JA gain not significant | `README.md` | Corrected |
| 32 | BM25 tie order (JA/ZH) | BM25 | Documented (tie range recorded) |
| 33 | CmedqaRetrieval subsample, ChatDoctor as MRR | benchmarks | Rerun: full corpus, nDCG@10 |
| 34 | Two more bugs in v1 script | `v1_original/run_experiment1.sh` | Documented, not fixed (provenance) |
| 35 | `run_medweb_today.sh` RRF column empty | `run_medweb_today.sh` | Script removed |
| 36 | Per-query BEIR and label-free values not saved | `camera_ready_extras.py` | Fixed for future runs |
| 37 | Figure 1 and prior claims not supported | paper | Stated in the paper |
| 38 | Gibbs sampler seeds only its first chunk | `lex_tm_model.py` | Open (`docs/GIBBS_DETERMINISM_FIX.md`) |
