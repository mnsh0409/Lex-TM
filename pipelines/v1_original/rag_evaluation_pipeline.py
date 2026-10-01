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
    def __init__(self, dense_model_name: str, lex_tm_model, dictionary, language: str = 'en', min_query_tokens: int = 0):
        print(f"Initializing LexTMRouter with dense model: {dense_model_name}")
        self.dense_model = SentenceTransformer(dense_model_name)
        self.lex_tm_model = lex_tm_model
        self.dictionary = dictionary        # UniversalRAGLoader instance
        self.language = language            # 'en' or 'zh'
 
        # Corpus indexing state — all three lists are index-aligned
        self.corpus_texts: List[str] = []       # raw text for dense encoding
        self.corpus_chunk_ids: List[str] = []   # chunk-level unique ID
        self.corpus_doc_ids: List[str] = []     # parent document ID
        self.dense_embeddings = None
        self.topic_distributions = None
        self.min_query_tokens = min_query_tokens
 
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
        if emb_cache_path and os.path.exists(emb_cache_path):
            print(f"  Loading cached embeddings from {emb_cache_path}")
            self.dense_embeddings = torch.tensor(
                np.load(emb_cache_path), dtype=torch.float32
            )
        else:
            print("  Computing dense embeddings (BGE-m3)...")
            self.dense_embeddings = self.dense_model.encode(
                self.corpus_texts,
                batch_size=256,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
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
            for text in tqdm(self.corpus_texts, desc="Inferring Topics", unit="chunk"):
                bow     = self._get_bow(text)
                theta_d = self.lex_tm_model.get_document_topic_distribution(bow)
                topic_dists.append(theta_d)
            self.topic_distributions = np.array(topic_dists)
 
            if tdist_cache_path:
                np.save(tdist_cache_path, self.topic_distributions)
                print(f"  Topic distributions cached to {tdist_cache_path}")
 
        print("Indexing complete.")
 
    def retrieve(self, query: str, top_k: int = 10, gamma: float = 0.7) -> List[Dict]:
        """
        Retrieves the top-k chunks via soft-weighted interpolation of dense
        cosine similarity and Lex-TM Jensen-Shannon topic similarity.
 
        Returns
        -------
        List of dicts, each containing:
            chunk_id    : str   — unique chunk identifier
            doc_id      : str   — parent document identifier
            text        : str   — raw chunk text
            final_score : float — interpolated retrieval score
            dense_score : float — cosine similarity component
            topic_score : float — 1 - JSD component
        """
        # --- Query representations ---
        query_dense = self.dense_model.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )
        query_bow   = self._get_bow(query)
        # Query-side gating: fall back to pure dense when BoW is too sparse
        # for reliable topic distribution estimation
        effective_gamma = (
            1.0 if (self.min_query_tokens > 0
                and len(query_bow) < self.min_query_tokens)
            else gamma
        )
        query_theta = self.lex_tm_model.get_document_topic_distribution(query_bow)
        query_theta = np.expand_dims(query_theta, axis=0)
 
        # --- Dense cosine similarity ---
        # Normalised embeddings → dot product == cosine similarity
        dense_scores = torch.matmul(self.dense_embeddings, query_dense).cpu().numpy()
        dense_scores = (dense_scores + 1.0) / 2.0      # scale [-1,1] → [0,1]
 
        # --- Topic similarity (inverse JSD) ---
        jsd_distances = jensenshannon(self.topic_distributions, query_theta, axis=1)
        topic_scores  = 1.0 - jsd_distances
 
        # --- Interpolation ---
        # Score(q, d) = γ · Cosine(Eq, Ed) + (1 − γ) · (1 − JSD(θq ∥ θd))
        final_scores = (effective_gamma * dense_scores) + ((1.0 - effective_gamma) * topic_scores)
 
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

class HybridRouter(LexTMRouter):
    """
    Extends LexTMRouter with BM25+Dense reciprocal rank fusion baseline.
    RRF score: 1/(k+rank_dense) + 1/(k+rank_bm25), k=60 standard.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25_index = None

    def index_corpus(self, chunk_dicts, cache_path=None):
        super().index_corpus(chunk_dicts, cache_path)
        # Build BM25 index from pre-tokenised clean tokens
        tokenised = [
            self.dictionary.clean_and_tokenize(c['text'], self.language)
            for c in chunk_dicts
        ]
        self.bm25_index = BM25Okapi(tokenised)
        print("  BM25 index built.")

    def retrieve_rrf(self, query: str, top_k: int = 10, k: int = 60):
        """Reciprocal rank fusion of dense and BM25 rankings."""
        # Dense ranking
        dense_results = self.retrieve(query, top_k=len(self.corpus_texts),
                                      gamma=1.0)
        dense_ranks = {r['chunk_id']: i+1 for i, r in enumerate(dense_results)}

        # BM25 ranking
        query_tokens = self.dictionary.clean_and_tokenize(query, self.language)
        bm25_scores  = self.bm25_index.get_scores(query_tokens)
        bm25_order   = np.argsort(bm25_scores)[::-1]
        bm25_ranks   = {self.corpus_chunk_ids[idx]: i+1
                        for i, idx in enumerate(bm25_order)}

        # RRF fusion
        all_ids = set(dense_ranks) | set(bm25_ranks)
        rrf = {
            cid: 1/(k + dense_ranks.get(cid, len(all_ids))) +
                 1/(k + bm25_ranks.get(cid, len(all_ids)))
            for cid in all_ids
        }
        top_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

        return [
            {
                'chunk_id':    cid,
                'doc_id':      self.corpus_doc_ids[
                                   self.corpus_chunk_ids.index(cid)],
                'text':        self.corpus_texts[
                                   self.corpus_chunk_ids.index(cid)],
                'final_score': rrf[cid],
            }
            for cid in top_ids
        ]

 
def evaluate_retrieval(router: LexTMRouter, evaluation_dataset: List[Dict], gamma: float = 0.7):
    """
    Calculates MRR and Recall@K across the evaluation set.
 
    Ground truth matching is now done by chunk_id (not raw text string).
    This eliminates silent misses caused by whitespace normalisation
    differences between indexed text and ground truth text.
 
    The evaluation_dataset must be built with chunk_ids as ground truth.
    See build_evaluation_dataset() in main_pipeline.py for the updated
    construction logic.
 
    Each item in evaluation_dataset must have:
        query              : str        — the natural language query
        ground_truth_ids   : set[str]   — set of valid ground truth chunk_ids
        ground_truth_doc_id: str        — parent doc_id (for doc-level metrics)
    """
    recall_at_5, recall_at_10, recall_at_20 = [], [], []
    reciprocal_ranks = []
 
    for item in tqdm(evaluation_dataset, desc=f"Evaluating (Gamma={gamma})", unit="query"):
        query               = item['query']
        gt_chunk_ids        = item['ground_truth_ids']          # set of chunk_id strings
        gt_doc_id           = item.get('ground_truth_doc_id')   # optional doc-level check
 
        retrieved   = router.retrieve(query, top_k=20, gamma=gamma)
        ret_chunk_ids = [r['chunk_id'] for r in retrieved]
        ret_doc_ids   = [r['doc_id']   for r in retrieved]
 
        # --- MRR: rank of first relevant chunk ---
        rank = next(
            (i + 1 for i, cid in enumerate(ret_chunk_ids) if cid in gt_chunk_ids),
            0
        )
        reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
 
        # --- Recall@K (chunk level) ---
        recall_at_5.append( 1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:5])  else 0)
        recall_at_10.append(1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:10]) else 0)
        recall_at_20.append(1 if any(cid in gt_chunk_ids for cid in ret_chunk_ids[:20]) else 0)
 
    print(f"--- Extrinsic Retrieval Evaluation (Gamma = {gamma}) ---")
    print(f"MRR:         {np.mean(reciprocal_ranks):.4f}")
    print(f"Recall@5:    {np.mean(recall_at_5):.4f}")
    print(f"Recall@10:   {np.mean(recall_at_10):.4f}")
    print(f"Recall@20:   {np.mean(recall_at_20):.4f}")
 
def bootstrap_mrr_ci(
    router: LexTMRouter,
    evaluation_dataset: list,
    gamma: float = 0.7,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Computes bootstrapped confidence interval for MRR.
    No re-running of retrieval needed — uses cached per-query RR values.
    """
    import random
    random.seed(seed)

    # Collect per-query reciprocal ranks
    rr_scores = []
    for item in evaluation_dataset:
        retrieved = router.retrieve(item['query'], top_k=20, gamma=gamma)
        ret_chunk_ids = [r['chunk_id'] for r in retrieved]
        gt = item['ground_truth_ids']
        rank = next((i+1 for i, cid in enumerate(ret_chunk_ids)
                     if cid in gt), 0)
        rr_scores.append(1.0/rank if rank > 0 else 0.0)

    n = len(rr_scores)
    bootstrap_mrrs = []
    for _ in range(n_bootstrap):
        sample = random.choices(rr_scores, k=n)
        bootstrap_mrrs.append(sum(sample) / n)

    bootstrap_mrrs.sort()
    alpha = (1 - ci) / 2
    lower = bootstrap_mrrs[int(alpha * n_bootstrap)]
    upper = bootstrap_mrrs[int((1-alpha) * n_bootstrap)]
    observed_mrr = sum(rr_scores) / n

    return {
        'mrr': observed_mrr,
        'ci_lower': lower,
        'ci_upper': upper,
        'n_queries': n,
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
