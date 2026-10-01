"""
rag_evaluation_pipeline_patched.py
===================================
Drop-in replacement for rag_evaluation_pipeline.py.

Fixes vs original
-----------------
FIX 1 — evaluate_retrieval now RETURNS rr_scores (List[float]) instead of None.
         This lets the caller pass scores directly to bootstrap_mrr_ci without
         re-running retrieval.

FIX 2 — bootstrap_mrr_ci now accepts pre-computed rr_scores (matching the
         validation pipeline API). No retrieval is re-run. Uses numpy RNG with
         fixed seed for reproducibility (original used random.seed which only
         controls Python stdlib, not numpy).

FIX 3 — HybridRouter builds a _chunk_id_to_idx dict at index time so that
         retrieve_rrf no longer calls corpus_chunk_ids.index(cid) (O(N) per
         result). Now O(1) per result — critical for large corpora.

FIX 4 — main_pipeline1.py wires --run_bm25 to HybridRouter. See the companion
         patch main_pipeline1_patched.py.

Everything else (k=60, tokeniser routing, cache logic, JSD formula) is
identical to the original and to rag_evaluation_pipeline_valid.py.
"""

import os
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from tqdm import tqdm
from rank_bm25 import BM25Okapi


class LexTMRouter:
    """
    Semantic routing layer that interpolates continuous dense vector similarity
    with Lex-TM topic distribution similarity.
    """
    def __init__(self, dense_model_name: str, lex_tm_model, dictionary,
                 language: str = 'en', min_query_tokens: int = 0,
                 query_smoothing: float = 0.1, adaptive_gamma: bool = True,
                 query_prompt_name: str = None, embed_batch_size: int = 256,
                 model_dtype: str = None, use_safetensors: bool = True,
                 trust_remote_code: bool = False):
        print(f"Initializing LexTMRouter with dense model: {dense_model_name}")
        # model_dtype: pass 'float16' for large (8B) models to fit VRAM.
        #
        # use_safetensors=True (default): forces from_pretrained to load the
        # .safetensors weights and refuse the legacy pytorch_model.bin path.
        # This sidesteps CVE-2025-32434: transformers>=4.48 blocks torch.load
        # of .bin weights unless torch>=2.6 ("Due to a serious vulnerability
        # ... upgrade torch to at least v2.6"). The restriction does NOT apply
        # to safetensors, so forcing it lets the current torch load the model
        # without a risky global torch upgrade or disabling the safety check.
        # Set use_safetensors=False only for a model that genuinely ships no
        # safetensors weights (then you must upgrade torch>=2.6 instead).
        #
        # trust_remote_code=False (default, deliberately opt-in): some models
        # (e.g. Snowflake/snowflake-arctic-embed-*-v2.0) ship custom modeling
        # code and REQUIRE trust_remote_code=True to load at all -- but this
        # executes arbitrary Python from the model repo, so it is never
        # enabled by default. Only set it True for a specific model you have
        # verified needs it and whose source you trust (check the model card).
        model_kwargs = {}
        if model_dtype is not None:
            model_kwargs["torch_dtype"] = model_dtype
        if use_safetensors:
            model_kwargs["use_safetensors"] = True
        st_kwargs = {"model_kwargs": model_kwargs} if model_kwargs else {}
        if trust_remote_code:
            st_kwargs["trust_remote_code"] = True
        self.dense_model = SentenceTransformer(dense_model_name, **st_kwargs)
        self.dense_model_name = dense_model_name
        self.lex_tm_model = lex_tm_model
        self.dictionary = dictionary
        self.language = language

        self.corpus_texts: List[str] = []
        self.corpus_chunk_ids: List[str] = []
        self.corpus_doc_ids: List[str] = []
        self.dense_embeddings = None
        self.topic_distributions = None
        self.min_query_tokens  = min_query_tokens
        # FIX 3: smoothing weight toward uniform prior for short queries (0 = off)
        self.query_smoothing   = query_smoothing
        # FIX 4: scale gamma down when query topic distribution is confident
        self.adaptive_gamma    = adaptive_gamma
        # Instruction-based models (Qwen3-Embedding) require a query-side prompt.
        # BGE-m3 is NOT instruction-based: leave this None for BGE-m3.
        # Documents are always encoded WITHOUT a prompt (asymmetric retrieval).
        self.query_prompt_name = query_prompt_name
        self.embed_batch_size  = embed_batch_size

    def _encode_query(self, query: str):
        """
        Encode a single query. For instruction-based models (Qwen3-Embedding)
        this applies the model's registered 'query' prompt; for BGE-m3
        (query_prompt_name=None) it encodes the raw text. Documents never
        receive the prompt, preserving the intended asymmetric-retrieval setup.

        BUGFIX: some encoders (e.g. Qwen3-Embedding checkpoints) load in
        bfloat16 by default when no --model_dtype override is given. numpy
        has no native bfloat16 support, so any later `.numpy()` call on a
        bf16 tensor raises `TypeError: Got unsupported ScalarType BFloat16`.
        Casting to float32 immediately after encoding makes every downstream
        op (matmul, np.save, cosine similarity) dtype-safe regardless of what
        precision the underlying model was loaded in.
        """
        enc_kwargs = dict(convert_to_tensor=True, normalize_embeddings=True)
        if self.query_prompt_name is not None:
            enc_kwargs["prompt_name"] = self.query_prompt_name
        return self.dense_model.encode(query, **enc_kwargs).float()

    def _get_bow(self, text: str) -> List[Tuple[int, int]]:
        """
        Converts raw text into a Bag-of-Words using the SAME tokenizer as
        training (clean_and_tokenize), not naive whitespace split.
        """
        tokens = self.dictionary.clean_and_tokenize(text, self.language)
        word_counts = {}
        for token in tokens:
            if token in self.dictionary.token2id:
                word_id = self.dictionary.token2id[token]
                word_counts[word_id] = word_counts.get(word_id, 0) + 1
        return list(word_counts.items())

    def index_corpus(self, chunk_dicts: List[Dict], cache_path: str = None,
                     emb_cache_path: str = None, tdist_cache_path: str = None):
        """
        Cache paths.
        - Preferred: pass emb_cache_path and tdist_cache_path SEPARATELY.
          Embeddings are model-specific; topic distributions are shared across
          dense encoders (they never see embeddings). Keeping them separate
          lets a Qwen run reuse BGE-era topic distributions while forcing a
          fresh embedding computation — and prevents loading the wrong vectors.
        - Backward-compatible: if only cache_path is given, both files share
          that prefix (original single-model behaviour).
        """
        print(f"Indexing {len(chunk_dicts)} document chunks...")

        self.corpus_texts     = [c['text']     for c in chunk_dicts]
        self.corpus_chunk_ids = [c['chunk_id'] for c in chunk_dicts]
        self.corpus_doc_ids   = [c['doc_id']   for c in chunk_dicts]

        if emb_cache_path is None and cache_path is not None:
            emb_cache_path = f"{cache_path}_embeddings"
        if tdist_cache_path is None and cache_path is not None:
            tdist_cache_path = f"{cache_path}_topicdists"

        emb_file   = f"{emb_cache_path}.npy"   if emb_cache_path   else None
        tdist_file = f"{tdist_cache_path}.npy" if tdist_cache_path else None

        # Dense embeddings (model-specific)
        # DEVICE FIX: the two paths below must land the tensor on the SAME
        # device, otherwise a matmul against a query embedding (which follows
        # the model's device) raises "Expected all tensors to be on the same
        # device". The cached path (torch.tensor(np.load(...))) defaults to
        # CPU; the fresh-compute path (encode(convert_to_tensor=True)) defaults
        # to the model's device (GPU when present). We resolve the model
        # device once and move the embeddings onto it in BOTH branches, so
        # behaviour is identical whether or not a cache file exists.
        target_device = getattr(self.dense_model, "device", None)
        if target_device is None:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        if emb_file and os.path.exists(emb_file):
            print(f"  Loading cached embeddings from {emb_file}")
            self.dense_embeddings = torch.tensor(
                np.load(emb_file), dtype=torch.float32
            ).to(target_device)
        else:
            print(f"  Computing dense embeddings ({self.dense_model_name})...")
            # Documents encoded WITHOUT a query prompt (asymmetric retrieval).
            # .float() cast: see _encode_query() docstring for why this is
            # required (bfloat16 -> numpy is unsupported).
            self.dense_embeddings = self.dense_model.encode(
                self.corpus_texts,
                batch_size=self.embed_batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
            ).float().to(target_device)
            if emb_file:
                os.makedirs(os.path.dirname(emb_file) or '.', exist_ok=True)
                np.save(emb_file, self.dense_embeddings.cpu().numpy())
                print(f"  Embeddings cached to {emb_file}")

        # Topic distributions (model-independent, shared across encoders)
        if tdist_file and os.path.exists(tdist_file):
            print(f"  Loading cached topic distributions from {tdist_file}")
            self.topic_distributions = np.load(tdist_file)
        else:
            print("  Computing Lex-TM topic distributions...")
            topic_dists = []
            for text in tqdm(self.corpus_texts, desc="Inferring Topics", unit="chunk"):
                bow     = self._get_bow(text)
                theta_d = self.lex_tm_model.get_document_topic_distribution(bow)
                topic_dists.append(theta_d)
            self.topic_distributions = np.array(topic_dists)

            if tdist_file:
                os.makedirs(os.path.dirname(tdist_file) or '.', exist_ok=True)
                np.save(tdist_file, self.topic_distributions)
                print(f"  Topic distributions cached to {tdist_file}")

        print("Indexing complete.")

    def retrieve(self, query: str, top_k: int = 10, gamma: float = 0.7) -> List[Dict]:
        query_dense = self._encode_query(query)
        query_bow = self._get_bow(query)
        effective_gamma = (
            1.0 if (self.min_query_tokens > 0
                    and len(query_bow) < self.min_query_tokens)
            else gamma
        )
        query_theta = self.lex_tm_model.get_document_topic_distribution(query_bow)

        # FIX 3: smooth noisy short-query topic distributions toward the
        # uniform prior. For MedWeb queries (~12 tokens) a single Gibbs
        # inference pass is unreliable; blending with 1/K stabilises the
        # distribution without discarding the topic signal entirely.
        # smoothing=0.1 is a fixed constant — no new hyperparameter to tune.
        if self.query_smoothing > 0.0:
            uniform_prior = np.ones(len(query_theta)) / len(query_theta)
            query_theta   = ((1.0 - self.query_smoothing) * query_theta
                             + self.query_smoothing * uniform_prior)

        # FIX 4: adaptive gamma — scale down dense weight proportionally to
        # how concentrated (confident) the query topic distribution is.
        # High entropy (diffuse) → topic signal unreliable → stay near gamma.
        # Low entropy (concentrated) → strong topic signal → reduce gamma.
        # Clipped to [gamma - 0.2, gamma] so we never move more than 0.2
        # from the tuned value; safe to apply without re-tuning gamma.
        if self.adaptive_gamma and effective_gamma < 1.0:
            K           = len(query_theta)
            entropy     = float(-np.sum(query_theta * np.log(query_theta + 1e-10)))
            max_entropy = float(np.log(K))
            # topic_confidence: 1 = perfectly concentrated, 0 = uniform
            topic_confidence = 1.0 - (entropy / max_entropy)
            # reduce gamma by up to 0.2 when confidence is high
            effective_gamma  = float(np.clip(
                effective_gamma - 0.2 * topic_confidence,
                effective_gamma - 0.2,
                effective_gamma,
            ))

        query_theta = np.expand_dims(query_theta, axis=0)

        dense_scores = torch.matmul(self.dense_embeddings, query_dense).cpu().numpy()
        dense_scores = (dense_scores + 1.0) / 2.0

        jsd_distances = jensenshannon(self.topic_distributions, query_theta, axis=1)
        topic_scores  = 1.0 - jsd_distances

        final_scores = (effective_gamma * dense_scores) + ((1.0 - effective_gamma) * topic_scores)
        top_indices  = np.argsort(final_scores)[::-1][:top_k]

        return [
            {
                "chunk_id":    self.corpus_chunk_ids[idx],
                "doc_id":      self.corpus_doc_ids[idx],
                "text":        self.corpus_texts[idx],
                "final_score": float(final_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "topic_score": float(topic_scores[idx]),
            }
            for idx in top_indices
        ]


class HybridRouter(LexTMRouter):
    """
    Extends LexTMRouter with BM25+Dense Reciprocal Rank Fusion baseline.
    RRF score: 1/(k+rank_dense) + 1/(k+rank_bm25), k=60 (Cormack et al. 2009).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25_index = None
        # FIX 3: O(1) chunk_id → corpus index lookup (replaces O(N) .index() call)
        self._chunk_id_to_idx: Dict[str, int] = {}

    def index_corpus(self, chunk_dicts: List[Dict], cache_path: str = None,
                     emb_cache_path: str = None, tdist_cache_path: str = None):
        super().index_corpus(
            chunk_dicts,
            cache_path=cache_path,
            emb_cache_path=emb_cache_path,
            tdist_cache_path=tdist_cache_path,
        )

        # Build BM25 index using same tokeniser as training
        print("  Building BM25 index...")
        tokenised = [
            self.dictionary.clean_and_tokenize(c['text'], self.language)
            for c in chunk_dicts
        ]
        self.bm25_index = BM25Okapi(tokenised)
        print("  BM25 index built.")

        # FIX 3: pre-build O(1) lookup
        self._chunk_id_to_idx = {
            cid: i for i, cid in enumerate(self.corpus_chunk_ids)
        }

    def retrieve_bm25(self, query: str, top_k: int = 10, **kwargs) -> List[Dict]:
        """
        Pure BM25 retrieval — no dense, no topic model, no fusion.

        Added to answer the lexical-overlap question: "is Lex-TM's
        gain just lexical-set matching that BM25 would also capture?" This
        isolates the lexical baseline on the SAME corpus/tokenizer/eval as
        Lex-TM, so BM25 vs Lex-TM is an apples-to-apples comparison. Uses the
        identical clean_and_tokenize path as both Lex-TM's BoW and the RRF
        BM25 index, so no tokenizer confound.
        """
        query_tokens = self.dictionary.clean_and_tokenize(query, self.language)
        bm25_scores  = self.bm25_index.get_scores(query_tokens)
        top_indices  = np.argsort(bm25_scores)[::-1][:top_k]
        return [
            {
                "chunk_id":    self.corpus_chunk_ids[idx],
                "doc_id":      self.corpus_doc_ids[idx],
                "text":        self.corpus_texts[idx],
                "final_score": float(bm25_scores[idx]),
            }
            for idx in top_indices
        ]

    def retrieve_rrf(self, query: str, top_k: int = 10, k: int = 60) -> List[Dict]:
        """
        Reciprocal rank fusion of dense and BM25 rankings.
        k=60 matches the paper statement and the validation pipeline.
        Dense encoder is whatever model the router was initialised with;
        query-side prompting (if any) is handled by _encode_query.
        """
        n = len(self.corpus_texts)

        # Dense ranking — raw dot products (normalised embeddings → cosine sim)
        query_dense  = self._encode_query(query)
        dense_scores = torch.matmul(self.dense_embeddings, query_dense).cpu().numpy()
        dense_order  = np.argsort(dense_scores)[::-1]
        dense_ranks  = np.empty(n, dtype=np.int32)
        dense_ranks[dense_order] = np.arange(1, n + 1)

        # BM25 ranking
        query_tokens = self.dictionary.clean_and_tokenize(query, self.language)
        bm25_scores  = self.bm25_index.get_scores(query_tokens)
        bm25_order   = np.argsort(bm25_scores)[::-1]
        bm25_ranks   = np.empty(n, dtype=np.int32)
        bm25_ranks[bm25_order] = np.arange(1, n + 1)

        # RRF fusion
        rrf_scores  = 1.0 / (k + dense_ranks) + 1.0 / (k + bm25_ranks)
        top_indices = np.argsort(rrf_scores)[::-1][:top_k]

        return [
            {
                # FIX 3: O(1) lookup via pre-built dict
                "chunk_id":    self.corpus_chunk_ids[idx],
                "doc_id":      self.corpus_doc_ids[idx],
                "text":        self.corpus_texts[idx],
                "final_score": float(rrf_scores[idx]),
            }
            for idx in top_indices
        ]


def evaluate_retrieval(
    router: LexTMRouter,
    evaluation_dataset: List[Dict],
    gamma: float = 0.7,
    label: str = "",
) -> List[float]:
    """
    Calculates MRR and Recall@K across the evaluation set.

    FIX 1: Now returns per-query reciprocal rank scores (List[float]) so the
    caller can pass them directly to bootstrap_mrr_ci without re-running
    retrieval. Original returned None.

    Parameters
    ----------
    label : str — optional label for the printed header (e.g. "val", "test").
    """
    recall_at_5, recall_at_10, recall_at_20 = [], [], []
    rr_scores: List[float] = []

    for item in tqdm(evaluation_dataset,
                     desc=f"Evaluating (Gamma={gamma})" + (f" [{label}]" if label else ""),
                     unit="query"):
        query        = item['query']
        gt_chunk_ids = item['ground_truth_ids']

        retrieved     = router.retrieve(query, top_k=20, gamma=gamma)
        ret_chunk_ids = [r['chunk_id'] for r in retrieved]

        rank = next(
            (i + 1 for i, cid in enumerate(ret_chunk_ids) if cid in gt_chunk_ids),
            0
        )
        rr_scores.append(1.0 / rank if rank > 0 else 0.0)

        recall_at_5.append( 1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:5])  else 0)
        recall_at_10.append(1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:10]) else 0)
        recall_at_20.append(1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:20]) else 0)

    header = f"Gamma = {gamma}" + (f" [{label}]" if label else "")
    print(f"--- Extrinsic Retrieval Evaluation ({header}, n={len(rr_scores)}) ---")
    print(f"MRR:         {np.mean(rr_scores):.4f}")
    print(f"Recall@5:    {np.mean(recall_at_5):.4f}")
    print(f"Recall@10:   {np.mean(recall_at_10):.4f}")
    print(f"Recall@20:   {np.mean(recall_at_20):.4f}")

    return rr_scores  # FIX 1


def evaluate_retrieval_rrf(
    router: HybridRouter,
    evaluation_dataset: List[Dict],
    k: int = 60,
) -> List[float]:
    """
    Evaluates BM25+Dense RRF baseline. Mirrors evaluate_retrieval_rrf()
    from rag_evaluation_pipeline_valid.py for numerical equivalence.
    Returns per-query reciprocal rank scores for bootstrap_mrr_ci().
    """
    recall_at_5, recall_at_10, recall_at_20 = [], [], []
    rr_scores: List[float] = []

    for item in tqdm(evaluation_dataset, desc="Evaluating RRF", unit="query"):
        query        = item['query']
        gt_chunk_ids = item['ground_truth_ids']

        retrieved     = router.retrieve_rrf(query, top_k=20, k=k)
        ret_chunk_ids = [r['chunk_id'] for r in retrieved]

        rank = next(
            (i + 1 for i, cid in enumerate(ret_chunk_ids) if cid in gt_chunk_ids),
            0
        )
        rr_scores.append(1.0 / rank if rank > 0 else 0.0)

        recall_at_5.append( 1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:5])  else 0)
        recall_at_10.append(1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:10]) else 0)
        recall_at_20.append(1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:20]) else 0)

    print(f"--- BM25+Dense RRF Evaluation (n={len(rr_scores)}) ---")
    print(f"MRR:         {np.mean(rr_scores):.4f}")
    print(f"Recall@5:    {np.mean(recall_at_5):.4f}")
    print(f"Recall@10:   {np.mean(recall_at_10):.4f}")
    print(f"Recall@20:   {np.mean(recall_at_20):.4f}")

    return rr_scores


def evaluate_retrieval_bm25(
    router: HybridRouter,
    evaluation_dataset: List[Dict],
) -> List[float]:
    """
    Evaluates the Pure BM25 baseline (no dense, no topic model). Same corpus,
    tokenizer, and evaluation set as Lex-TM, so Lex-TM vs BM25 is a clean
    apples-to-apples comparison. Directly addresses "is the gain just lexical
    matching?" Returns per-query RR scores for bootstrap_mrr_ci().
    """
    recall_at_5, recall_at_10, recall_at_20 = [], [], []
    rr_scores: List[float] = []

    for item in tqdm(evaluation_dataset, desc="Evaluating Pure BM25", unit="query"):
        query        = item['query']
        gt_chunk_ids = item['ground_truth_ids']

        retrieved     = router.retrieve_bm25(query, top_k=20)
        ret_chunk_ids = [r['chunk_id'] for r in retrieved]

        rank = next(
            (i + 1 for i, cid in enumerate(ret_chunk_ids) if cid in gt_chunk_ids),
            0
        )
        rr_scores.append(1.0 / rank if rank > 0 else 0.0)

        recall_at_5.append( 1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:5])  else 0)
        recall_at_10.append(1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:10]) else 0)
        recall_at_20.append(1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:20]) else 0)

    print(f"--- Pure BM25 Evaluation (n={len(rr_scores)}) ---")
    print(f"MRR:         {np.mean(rr_scores):.4f}")
    print(f"Recall@5:    {np.mean(recall_at_5):.4f}")
    print(f"Recall@10:   {np.mean(recall_at_10):.4f}")
    print(f"Recall@20:   {np.mean(recall_at_20):.4f}")

    return rr_scores


def bootstrap_mrr_ci(
    rr_scores: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bootstrapped confidence interval for MRR from pre-computed per-query RR scores.

    FIX 2: Takes rr_scores directly — no retrieval re-running. Matches the API
    and numpy RNG of rag_evaluation_pipeline_valid.py for identical CI values.

    Original signature was bootstrap_mrr_ci(router, evaluation_dataset, gamma, ...)
    which re-ran all retrieval for every bootstrap call (slow and wasteful).
    """
    rng    = np.random.default_rng(seed)
    scores = np.array(rr_scores)
    n      = len(scores)

    bootstrap_mrrs = np.array([
        rng.choice(scores, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(bootstrap_mrrs, alpha * 100))
    upper = float(np.percentile(bootstrap_mrrs, (1.0 - alpha) * 100))
    mrr   = float(scores.mean())

    print(f"MRR = {mrr:.4f} [{ci*100:.0f}% CI: {lower:.4f}–{upper:.4f}] "
          f"(n={n}, B={n_bootstrap})")

    return {
        "mrr": mrr, "ci_lower": lower, "ci_upper": upper,
        "n": n, "n_bootstrap": n_bootstrap,
    }


def paired_bootstrap_diff(
    rr_a: List[float],
    rr_b: List[float],
    label_a: str = "A",
    label_b: str = "B",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Honest paired bootstrap for "does A actually beat B?"

    Unlike comparing two independently-computed CIs (conservative and low-power
    — overlap does not mean parity, it means the test couldn't distinguish
    them), this resamples query INDICES once per draw and applies the SAME
    resampled indices to both rr_a and rr_b, then computes the per-draw
    difference (mean(A[idx]) - mean(B[idx])). This directly tests the paired
    difference, which is the statistically correct comparison when both
    methods are evaluated on the identical query set.

    Requires rr_a and rr_b to be the same length and in the same query order
    — i.e. computed from the same eval_dataset in the same pipeline run.
    This holds automatically when --run_bm25 is passed to
    main_pipeline1_patched.py, since evaluate_retrieval() and
    evaluate_retrieval_rrf() both iterate the same eval_dataset once.

    Reports the CI on the DIFFERENCE. If that interval excludes 0, the
    difference is significant at the given confidence level. If it includes
    0 — including if A's point estimate happens to be higher than B's — the
    honest conclusion is that the data does not support a "beats" claim.
    This function will not manufacture a win: if A's scores are genuinely
    not higher than B's, the reported CI will straddle zero, as it should.
    """
    a = np.array(rr_a)
    b = np.array(rr_b)
    if len(a) != len(b):
        raise ValueError(
            f"paired_bootstrap_diff requires equal-length, same-query-order "
            f"arrays (got {len(a)} vs {len(b)}) — rr_a and rr_b must come "
            f"from the same eval_dataset in the same run."
        )
    n = len(a)
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)   # SAME indices applied to both arms
        diffs[i] = a[idx].mean() - b[idx].mean()

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(diffs, alpha * 100))
    upper = float(np.percentile(diffs, (1.0 - alpha) * 100))
    point_diff = float(a.mean() - b.mean())
    significant = not (lower <= 0.0 <= upper)

    verdict = (
        f"significant: {label_a} beats {label_b}" if (significant and point_diff > 0) else
        f"significant: {label_b} beats {label_a}" if (significant and point_diff < 0) else
        "NOT significant — CI on the difference includes 0"
    )

    print(f"--- Paired bootstrap: {label_a} vs {label_b} (n={n}, B={n_bootstrap}) ---")
    print(f"  {label_a} MRR: {a.mean():.4f}   {label_b} MRR: {b.mean():.4f}")
    print(f"  Difference ({label_a} - {label_b}): {point_diff:+.4f} "
          f"[{ci*100:.0f}% CI: {lower:+.4f} to {upper:+.4f}]")
    print(f"  Verdict: {verdict}")

    return {
        "label_a": label_a, "label_b": label_b,
        "mrr_a": float(a.mean()), "mrr_b": float(b.mean()),
        "diff": point_diff, "diff_ci_lower": lower, "diff_ci_upper": upper,
        "significant": significant, "n": n, "n_bootstrap": n_bootstrap,
    }


# Unchanged from original
def get_llm_judge_prompt(query: str, retrieved_context: str, generated_answer: str) -> str:
    prompt = f"""You are an expert aviation analyst grading an AI assistant's response.
Evaluate the 'Generated Answer' based STRICTLY on the 'Retrieved Context'.

[Query]: {query}
[Retrieved Context]: {retrieved_context}
[Generated Answer]: {generated_answer}

You must evaluate the response on two dimensions. Provide your final output as a strict JSON object.

1. Factual Fidelity (Boolean 0 or 1)
- Score 1: The Generated Answer contains NO external hallucinations. Every technical claim made is explicitly supported by the Retrieved Context.
- Score 0: The Generated Answer introduces facts, procedures, or technical jargon not found in the Retrieved Context, or directly contradicts the context.

2. Contextual Utility (Ordinal 1-5)
- 1: The response fails to answer the query or the provided context was entirely irrelevant.
- 3: The response partially answers the query, but misses key technical granularity requested.
- 5: The response thoroughly and accurately answers the query utilizing highly specific technical details from the context.

Output Format:
{{
  "Factual_Fidelity": <0 or 1>,
  "Contextual_Utility": <1 to 5>,
  "Reasoning": "<A brief one-sentence justification>"
}}
"""
    return prompt
