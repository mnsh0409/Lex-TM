import os
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

class LexTMRouter:
    """
    Semantic routing layer that interpolates continuous dense vector similarity
    with Lex-TM topic distribution similarity.
    """
    def __init__(self, dense_model_name: str, lex_tm_model, dictionary,
                 language: str = 'en', min_query_tokens: int = 0):
        print(f"Initializing LexTMRouter with dense model: {dense_model_name}")
        self.dense_model = SentenceTransformer(dense_model_name)
        self.lex_tm_model = lex_tm_model
        self.dictionary = dictionary
        self.language = language

        # Query-side gating: queries with fewer BoW tokens than this threshold
        # fall back to pure dense (effective_gamma=1.0). Set to 0 to disable.
        self.min_query_tokens = min_query_tokens

        # Corpus indexing state — all lists are index-aligned
        self.corpus_texts: List[str] = []
        self.corpus_chunk_ids: List[str] = []
        self.corpus_doc_ids: List[str] = []
        self.dense_embeddings = None
        self.topic_distributions = None

        # BM25 index — built in index_corpus() when run_bm25=True
        self.bm25_index = None
        # O(1) chunk_id → corpus index lookup for RRF
        self._chunk_id_to_idx: Dict[str, int] = {}

    def _get_bow(self, text: str) -> List[Tuple[int, int]]:
        """
        Converts raw text into a Bag-of-Words using the SAME tokenizer that
        was used to build the vocabulary during training.

        FIX (tokenizer mismatch): The original used `text.lower().split()`
        (naive whitespace split), which differs from the training pipeline
        (NLTK + lemmatization for English, jieba for Chinese). Most query
        tokens were OOV, producing a near-uniform query_theta and making the
        JSD routing signal meaningless. Now routes through
        `self.dictionary.clean_and_tokenize(text, self.language)` — the exact
        same method called in UniversalRAGLoader.process_corpus().
        """
        tokens = self.dictionary.clean_and_tokenize(text, self.language)
        word_counts = {}
        for token in tokens:
            if token in self.dictionary.token2id:
                word_id = self.dictionary.token2id[token]
                word_counts[word_id] = word_counts.get(word_id, 0) + 1
        return list(word_counts.items())

    def index_corpus(self, chunk_dicts: List[Dict], cache_path: str = None):
        """
        Precomputes and caches dense embeddings and Lex-TM topic distributions
        for the entire corpus.

        SPEED-UP 3 (Disk cache): Dense embeddings and topic distributions are
        saved to disk on the first run and loaded on subsequent runs. This
        eliminates redundant BGE-m3 encoding (15–30 min on GPU per 100K chunks)
        and topic distribution computation across ablation runs that share the
        same corpus and Lex-TM model.

        Cache key design: cache_path should encode everything that affects the
        cached values — dataset_name, tau, K — but NOT gamma, because gamma
        only affects the interpolation in retrieve() and does not change the
        embeddings or topic distributions. This means all gamma ablation runs
        for the same (dataset, tau, K) share one cache.

        In main_pipeline.py, pass:
            cache_path = (
                f"experiment_logs/cache_{args.dataset_name}"
                f"_tau{args.tau}_K{args.num_topics}"
            )
            router.index_corpus(all_chunks_dicts, cache_path=cache_path)

        Parameters
        ----------
        chunk_dicts  : list of dict  — keys: 'text', 'chunk_id', 'doc_id'
        cache_path   : str or None   — base path for cache files; None disables
                       caching. Two files are written:
                           <cache_path>_embeddings.npy
                           <cache_path>_topicdists.npy
        """
        print(f"Indexing {len(chunk_dicts)} document chunks...")

        self.corpus_texts     = [c['text']     for c in chunk_dicts]
        self.corpus_chunk_ids = [c['chunk_id'] for c in chunk_dicts]
        self.corpus_doc_ids   = [c['doc_id']   for c in chunk_dicts]

        emb_cache_path   = f"{cache_path}_embeddings.npy"  if cache_path else None
        tdist_cache_path = f"{cache_path}_topicdists.npy"  if cache_path else None

        # ------------------------------------------------------------------
        # 1. Dense embeddings — load from cache or compute and save
        # ------------------------------------------------------------------
        # PATCH (device fix, added post-hoc during later debugging):
        # SentenceTransformer.encode() places tensors on self.dense_model.device
        # automatically, but torch.tensor(np.load(...)) always defaults to CPU
        # regardless of that device. On a GPU machine, a cache-load run then
        # mismatches against the GPU-encoded query at retrieve()/retrieve_rrf()
        # time (RuntimeError: mat is on cpu, different from other tensors on
        # cuda:0). Explicit .to() on both branches makes the device consistent
        # regardless of which branch executes.
        if emb_cache_path and os.path.exists(emb_cache_path):
            print(f"  Loading cached embeddings from {emb_cache_path}")
            self.dense_embeddings = torch.tensor(
                np.load(emb_cache_path), dtype=torch.float32
            ).to(self.dense_model.device)
        else:
            print("  Computing dense embeddings (BGE-m3)...")
            self.dense_embeddings = self.dense_model.encode(
                self.corpus_texts,
                batch_size=256,
                convert_to_tensor=True,
                normalize_embeddings=True,
            ).to(self.dense_model.device)
            if emb_cache_path:
                os.makedirs(os.path.dirname(emb_cache_path) or '.', exist_ok=True)
                np.save(emb_cache_path, self.dense_embeddings.cpu().numpy())
                print(f"  Embeddings cached to {emb_cache_path}")

        # ------------------------------------------------------------------
        # 2. Topic distributions — load from cache or compute and save
        # Note: topic distributions depend on the Lex-TM model (tau, K) but
        # NOT on gamma. Caching here saves computation across all gamma runs.
        # ------------------------------------------------------------------
        if tdist_cache_path and os.path.exists(tdist_cache_path):
            print(f"  Loading cached topic distributions from {tdist_cache_path}")
            self.topic_distributions = np.load(tdist_cache_path)
        else:
            print("  Computing Lex-TM topic distributions...")
            topic_dists = []
            for text in self.corpus_texts:
                bow     = self._get_bow(text)
                theta_d = self.lex_tm_model.get_document_topic_distribution(bow)
                topic_dists.append(theta_d)
            self.topic_distributions = np.array(topic_dists)

            if tdist_cache_path:
                np.save(tdist_cache_path, self.topic_distributions)
                print(f"  Topic distributions cached to {tdist_cache_path}")

        print("Indexing complete.")

        # O(1) chunk_id → index lookup used by retrieve_rrf()
        self._chunk_id_to_idx = {
            cid: i for i, cid in enumerate(self.corpus_chunk_ids)
        }

    def _build_bm25_index(self):
        """
        Builds a BM25 index from the indexed corpus using the same
        language-appropriate tokeniser as training. Called from
        index_corpus() when run_bm25=True, or on demand.

        Uses clean_and_tokenize() so Chinese/Japanese text is segmented
        with jieba/MeCab rather than whitespace — critical for correctness.
        """
        if not _BM25_AVAILABLE:
            raise ImportError(
                "rank_bm25 not installed. Run: pip install rank-bm25"
            )
        print("  Building BM25 index...")
        tokenised = [
            self.dictionary.clean_and_tokenize(text, self.language)
            for text in self.corpus_texts
        ]
        self.bm25_index = BM25Okapi(tokenised)
        print(f"  BM25 index built ({len(self.corpus_texts):,} documents).")

    def retrieve(self, query: str, top_k: int = 10, gamma: float = 0.7) -> List[Dict]:
        """
        Retrieves top-k chunks via soft-weighted interpolation of dense
        cosine similarity and Lex-TM JSD topic similarity.

        Query-side gating: if len(BoW(query)) < self.min_query_tokens,
        effective_gamma is set to 1.0 (pure dense fallback). This prevents
        noisy topic distributions from sparse short queries (e.g. factoid
        IR queries on MIRACL/Mr. TyDi where median BoW = 1–3 tokens)
        from penalising correct retrievals.
        """
        query_dense = self.dense_model.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )
        query_bow   = self._get_bow(query)

        # Query-side gating
        effective_gamma = (
            1.0 if (self.min_query_tokens > 0
                    and len(query_bow) < self.min_query_tokens)
            else gamma
        )

        query_theta = self.lex_tm_model.get_document_topic_distribution(query_bow)
        query_theta = np.expand_dims(query_theta, axis=0)

        # PATCH (device fix): defensive alignment in case dense_embeddings
        # and query_dense ever end up on different devices.
        query_dense = query_dense.to(self.dense_embeddings.device)
        dense_scores = torch.matmul(self.dense_embeddings, query_dense).cpu().numpy()
        dense_scores = (dense_scores + 1.0) / 2.0

        jsd_distances = jensenshannon(self.topic_distributions, query_theta, axis=1)
        topic_scores  = 1.0 - jsd_distances

        final_scores = (effective_gamma * dense_scores) + \
                       ((1.0 - effective_gamma) * topic_scores)

        top_indices = np.argsort(final_scores)[::-1][:top_k]

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

    def retrieve_rrf(self, query: str, top_k: int = 10, k: int = 60) -> List[Dict]:
        """
        BM25 + Dense Reciprocal Rank Fusion baseline.

        RRF(q,d) = 1/(k + rank_dense(d)) + 1/(k + rank_bm25(d))
        k=60 is the standard constant from Cormack et al. (2009).

        This is a parameter-free baseline — no gamma or tau to tune.
        Requires self.bm25_index to be built (call _build_bm25_index() first).
        """
        if self.bm25_index is None:
            raise RuntimeError(
                "BM25 index not built. Pass run_bm25=True to index_corpus() "
                "or call router._build_bm25_index() directly."
            )

        n = len(self.corpus_texts)

        # Dense ranking — full corpus, gamma=1.0
        query_dense  = self.dense_model.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )
        # PATCH (device fix): defensive alignment in case dense_embeddings
        # and query_dense ever end up on different devices.
        query_dense = query_dense.to(self.dense_embeddings.device)
        dense_scores = torch.matmul(self.dense_embeddings, query_dense).cpu().numpy()
        # rank 1 = best: argsort descending
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
        rrf_scores = 1.0 / (k + dense_ranks) + 1.0 / (k + bm25_ranks)
        top_indices = np.argsort(rrf_scores)[::-1][:top_k]

        return [
            {
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

    Returns per-query reciprocal rank scores so that bootstrap_mrr_ci()
    can resample them without re-running retrieval.

    Parameters
    ----------
    label : str  — optional label appended to the printed header line,
                   e.g. "val" or "test" for split-aware evaluation.

    Returns
    -------
    rr_scores : list of float  — reciprocal rank (0.0 if not found) per query
    """
    recall_at_5, recall_at_10, recall_at_20 = [], [], []
    rr_scores: List[float] = []

    for item in evaluation_dataset:
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

    return rr_scores


def evaluate_retrieval_rrf(
    router: LexTMRouter,
    evaluation_dataset: List[Dict],
    k: int = 60,
) -> List[float]:
    """
    Evaluates BM25+Dense RRF baseline. No gamma or tau parameters.
    Requires router.bm25_index to be built.
    Returns per-query reciprocal rank scores for bootstrap_mrr_ci().
    """
    recall_at_5, recall_at_10, recall_at_20 = [], [], []
    rr_scores: List[float] = []

    for item in evaluation_dataset:
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


def bootstrap_mrr_ci(
    rr_scores: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Computes a bootstrapped confidence interval for MRR from per-query
    reciprocal rank scores returned by evaluate_retrieval().

    No retrieval re-running — resamples the rr_scores list directly.
    Runs in milliseconds regardless of corpus size.

    Parameters
    ----------
    rr_scores   : per-query RR values from evaluate_retrieval()
    n_bootstrap : number of bootstrap resamples (default 1000)
    ci          : confidence level (default 0.95 → 95% CI)
    seed        : random seed for reproducibility

    Returns
    -------
    dict with keys: mrr, ci_lower, ci_upper, n_queries, n_bootstrap
    """
    rng = np.random.default_rng(seed)
    scores = np.array(rr_scores)
    n = len(scores)

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
        'mrr':         mrr,
        'ci_lower':    lower,
        'ci_upper':    upper,
        'n_queries':   n,
        'n_bootstrap': n_bootstrap,
    }


# --- Section 4.4: LLM-as-a-Judge Prompt Template ---

def get_llm_judge_prompt(query: str, retrieved_context: str, generated_answer: str) -> str:
    """
    Zero-shot deterministic prompt for evaluating downstream generative utility.
    """
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
