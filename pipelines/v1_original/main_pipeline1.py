import csv
import json
import os
import pandas as pd
import argparse
from rag_dataload import UniversalRAGLoader
from lex_tm_model import LexTMLdaModel
from rag_evaluation_pipeline import LexTMRouter, evaluate_retrieval
from typing import Dict, List, Tuple
 
 
# ==========================================
# QUERY COLUMN MAP
# Used only for datasets that use the title-as-query proxy.
# miracl_zh is NOT listed here — it uses real qrels via
# build_evaluation_dataset_miracl() and never needs a query column.
# ==========================================
QUERY_COLUMN = {
    'english_aviation': 'ReviewTitle',
    'chinese_aviation': 'ReviewTitle',
    'trec_covid':       'title',
    'nfcorpus':         'title',
    'thucnews':         'title',
    'sogou':            'contenttitle',
    'medweb_en':        'ReviewTitle',
    'medweb_ja':        'ReviewTitle',
    'medweb_zh':        'ReviewTitle',
}
 
# Maps dataset_name → language code for tokenisation routing
_DATASET_LANG: Dict[str, str] = {
    'english_aviation': 'en',
    'chinese_aviation': 'zh',
    'trec_covid':       'en',
    'nfcorpus':         'en',
    'thucnews':         'zh',
    'sogou':            'zh',
    'miracl_zh':        'zh',
    'mr_tydi_ja':       'ja',
    'mr_tydi_th':       'th',
    'medweb_en':        'en',
    'medweb_ja':        'ja',
    'medweb_zh':        'zh',
}
 
# Datasets that use human-annotated qrels rather than the title-as-query proxy.
# Values are the document_id prefix applied by the corresponding loader,
# which must match the prefix prepended to corpus_id in the qrels.
_QRELS_DATASETS: Dict[str, str] = {
    'miracl_zh':  'miracl_',
    'mr_tydi_ja': 'mr_tydi_ja_',
    'mr_tydi_th': 'mr_tydi_th_',
    'trec_covid': 'trec_covid_',
    'nfcorpus':   'nfcorpus_',
}
 
 
def build_evaluation_dataset(df: pd.DataFrame, chunks: list, dataset_type: str) -> list:
    """
    Generates Query -> Ground Truth pairs for extrinsic retrieval evaluation.
 
    Uses Title-to-Content retrieval as a proxy for natural RAG queries:
    the document title is the query; all chunks from ALL documents sharing
    that title are valid retrieval targets.
 
    TITLE GROUPING FIX: The original implementation treated each document
    independently — ground truth was the chunks of one specific document.
    When multiple documents share the same title (common in MedWeb where
    symptom combinations repeat across 2,560 posts), retrieving any other
    document with the same symptoms counted as a miss, suppressing MRR by
    a factor equal to the average group size.
 
    The fix groups documents by their title and treats all chunks from all
    documents with the same title as jointly valid ground truth. For datasets
    with unique titles (THUCNews, Sogou, aviation reviews), the grouping
    collapses to one-to-one and results are identical to the original.
    For MedWeb, where symptom combinations repeat, this corrects the bias.
 
    Each returned item contains:
        query               : str      — document title used as the query
        ground_truth_ids    : set[str] — chunk_ids from ALL docs with this title
        ground_truth_doc_id : str      — representative doc_id for the group
    """
    print(f"Building evaluation pairs for {dataset_type}...")
 
    query_col = QUERY_COLUMN.get(dataset_type)
    if query_col is None:
        raise ValueError(
            f"Unknown dataset_type '{dataset_type}'. "
            f"Add it to QUERY_COLUMN at the top of main_pipeline.py."
        )
 
    # Build doc_id -> set of chunk_ids
    chunk_lookup: Dict[str, set] = {}
    for chunk in chunks:
        doc_id = chunk['doc_id']
        if doc_id not in chunk_lookup:
            chunk_lookup[doc_id] = set()
        chunk_lookup[doc_id].add(chunk['chunk_id'])
 
    # Group documents by title — maps title -> (set of chunk_ids, rep doc_id)
    # All documents sharing a title contribute their chunks to one ground truth set
    title_groups: Dict[str, Dict] = {}
    skipped = 0
 
    for _, row in df.iterrows():
        doc_id = row['document_id']
        query  = str(row.get(query_col, '')).strip()
 
        if not query or doc_id not in chunk_lookup:
            skipped += 1
            continue
 
        if query not in title_groups:
            title_groups[query] = {
                'chunk_ids': set(),
                'rep_doc_id': doc_id,   # first doc seen for this title
            }
        title_groups[query]['chunk_ids'].update(chunk_lookup[doc_id])
 
    eval_data = [
        {
            'query':               title,
            'ground_truth_ids':    group['chunk_ids'],
            'ground_truth_doc_id': group['rep_doc_id'],
        }
        for title, group in title_groups.items()
    ]
 
    n_docs   = len(df) - skipped
    n_groups = len(eval_data)
    avg_size = n_docs / n_groups if n_groups > 0 else 0
 
    print(
        f"Generated {n_groups} evaluation pairs from {n_docs} documents "
        f"({skipped} skipped — no title or no chunks). "
        f"Avg docs per title group: {avg_size:.1f}"
    )
    if avg_size > 2.0:
        print(
            f"  NOTE: avg group size {avg_size:.1f} > 1 indicates duplicate "
            "titles — title grouping is active and materially affects results."
        )
 
    return eval_data
 
 
def load_queries_and_qrels(
    queries_path: str,
    qrels_path: str,
) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Loads MIRACL queries and qrels from the files produced by
    download_miracl_chinese.py.
 
    Parameters
    ----------
    queries_path : str  — path to queries.jsonl ({_id, text} per line)
    qrels_path   : str  — path to qrels/dev.tsv  (query_id, corpus_id, score)
 
    Returns
    -------
    queries : dict {query_id: query_text}
    qrels   : list of {query_id, corpus_id, score}
    """
    # Load queries
    queries: Dict[str, str] = {}
    with open(queries_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            queries[str(record['_id'])] = record['text']
 
    # Load qrels (tab-separated, no header)
    qrels: List[Dict] = []
    with open(qrels_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(
            f, fieldnames=['query_id', 'corpus_id', 'score'], delimiter='\t'
        )
        for row in reader:
            try:
                score_val = int(row['score'])
            except ValueError:
                continue # Safely skip the header row containing the word 'score'

            if score_val > 0:   # keep only positive relevance judgements
                qrels.append({
                    'query_id':  str(row['query_id']),
                    'corpus_id': str(row['corpus_id']),
                    'score':     int(row['score']),
                })
 
    print(f"Loaded {len(queries):,} queries and {len(qrels):,} qrel pairs.")
    return queries, qrels
 
 
def build_evaluation_dataset_qrels(
    queries: Dict[str, str],
    qrels: List[Dict],
    all_chunks_dicts: list,
    id_prefix: str,
) -> list:
    """
    Builds evaluation pairs from human-annotated qrels for any dataset that
    provides them — currently MIRACL Chinese, Mr. TyDi Japanese, Mr. TyDi Thai.
 
    Ground truth is a set of chunk_ids belonging to the qrel-referenced
    document, matching the interface expected by evaluate_retrieval().
 
    id_prefix must match the prefix used by the corresponding loader:
        'miracl_'      for miracl_zh  (load_miracl_zh)
        'mr_tydi_ja_'  for mr_tydi_ja (load_mr_tydi(..., 'ja'))
        'mr_tydi_th_'  for mr_tydi_th (load_mr_tydi(..., 'th'))
 
    Parameters
    ----------
    queries    : dict {query_id: query_text}
    qrels      : list of {query_id, corpus_id, score}
    all_chunks_dicts : list of chunk dicts from loader.process_corpus()
    id_prefix  : str  prefix to prepend to corpus_id when looking up chunks
 
    Returns
    -------
    list of {query, ground_truth_ids, ground_truth_doc_id}
    """
    print(f"Building qrels-based evaluation pairs (id_prefix='{id_prefix}')...")
 
    chunk_lookup: Dict[str, set] = {}
    for chunk in all_chunks_dicts:
        doc_id = chunk['doc_id']
        if doc_id not in chunk_lookup:
            chunk_lookup[doc_id] = set()
        chunk_lookup[doc_id].add(chunk['chunk_id'])
 
    eval_data = []
    skipped_no_query  = 0
    skipped_no_chunks = 0
 
    for qrel in qrels:
        query_id  = qrel['query_id']
        corpus_id = qrel['corpus_id']
        doc_id    = f'{id_prefix}{corpus_id}'
 
        query_text = queries.get(query_id, '').strip()
        if not query_text:
            skipped_no_query += 1
            continue
 
        if doc_id not in chunk_lookup:
            skipped_no_chunks += 1
            continue
 
        eval_data.append({
            'query':               query_text,
            'ground_truth_ids':    chunk_lookup[doc_id],
            'ground_truth_doc_id': doc_id,
        })
 
    print(
        f"Generated {len(eval_data):,} evaluation pairs "
        f"({skipped_no_query} skipped — query not found, "
        f"{skipped_no_chunks} skipped — document not in corpus)."
    )
    return eval_data
 
 
def run_end_to_end_pipeline(args):
    """
    Orchestrates all four phases of the Lex-TM RAG pipeline:
        1. Data loading and preprocessing
        2. Lex-TM training (Gibbs sampling with dynamic HHI prior)
        3. RAG corpus indexing and routing layer construction
        4. Extrinsic retrieval evaluation (MRR, Recall@K)
    """
    print(f"\n{'='*50}")
    print(f"Lex-TM Pipeline — {args.dataset_name.upper()}")
    print(f"  gamma={args.gamma} | tau={args.tau} | K={args.num_topics} | lambda={args.lambda_amp}")
    print(f"{'='*50}")
 
    # -------------------------------------------------------
    # PHASE 1: DATA LOADING & PREPROCESSING
    # -------------------------------------------------------
    print("\n=== Phase 1: Data Loading & Preprocessing ===")
 
    # FIX 3: Derive language once here; passed to both the loader and the router
    # so both use identical tokenisation throughout the full pipeline.
    lang = _DATASET_LANG.get(args.dataset_name, 'en')
 
    # chunk_size args allow per-dataset overrides via CLI.
    # MedWeb requires chunk_size=20 — texts average only 12 English words.
    # At the default chunk_size=150, min_chunk_len=30 would discard ~98%
    # of MedWeb documents. Run MedWeb with --chunk_size 20.
    loader = UniversalRAGLoader(
        chunk_size    = args.chunk_size,
        chunk_size_zh = args.chunk_size_zh if args.chunk_size_zh else args.chunk_size * 2,
        chunk_size_ja = args.chunk_size_ja if args.chunk_size_ja else args.chunk_size * 2,
        overlap       = max(1, args.chunk_size // 7),   # ~14% overlap, scales with chunk_size
        min_df        = 2,
    )
 
    loaders = {
        'trec_covid':       loader.load_beir,
        'nfcorpus':         loader.load_beir,
        'english_aviation': loader.load_english_aviation,
        'chinese_aviation': loader.load_chinese_aviation,
        'thucnews':         loader.load_thucnews,
        'sogou':            loader.load_sogou,
        'miracl_zh':        loader.load_miracl_zh,
        'mr_tydi_ja':       lambda fp: loader.load_mr_tydi(fp, 'ja'),
        'mr_tydi_th':       lambda fp: loader.load_mr_tydi(fp, 'th'),
        'trec_covid':       lambda fp: loader.load_beir(fp, prefix='trec_covid_'),
        'nfcorpus':         lambda fp: loader.load_beir(fp, prefix='nfcorpus_'),
        # MedWeb: one file, split by language at load time
        'medweb_en':        lambda fp: loader.load_medweb(fp, 'en'),
        'medweb_ja':        lambda fp: loader.load_medweb(fp, 'ja'),
        'medweb_zh':        lambda fp: loader.load_medweb(fp, 'zh'),
    }
    df = loaders[args.dataset_name](args.filepath)
 
    if df.empty:
        raise RuntimeError(
            f"DataFrame is empty after loading '{args.filepath}'. "
            "Check the file path and format."
        )
 
    # --max_docs: cap corpus size before chunking and vocabulary building.
    # Applied here so N_tokens — the dominant Gibbs sampler cost driver —
    # scales with the sample, not the full corpus. Uses a fixed seed so
    # results are reproducible and the sample can be cited explicitly:
    #   "General-domain evaluation used a stratified 10K-document sample
    #    of THUCNews (seed=42), disclosed to ensure reproducibility."
    if args.max_docs is not None and len(df) > args.max_docs:
        df = df.sample(n=args.max_docs, random_state=42).reset_index(drop=True)
        print(f"  Corpus capped at {args.max_docs:,} documents (random_state=42).")
    else:
        print(f"  Corpus size: {len(df):,} documents (no cap applied).")
 
    all_chunks_dicts, corpus_bow, doc_assignments = loader.process_corpus(df, language=lang)
 
    # -------------------------------------------------------
    # PHASE 2: LEX-TM TRAINING
    # -------------------------------------------------------
    print("\n=== Phase 2: Lex-TM Training ===")
 
    lex_tm = LexTMLdaModel(
        num_topics=args.num_topics,
        alpha=0.1,
        beta_base=0.01,
        tau=args.tau,
        lambda_amp=args.lambda_amp,
        random_state=42,
    )
 
    lex_tm.fit(
        corpus=corpus_bow,
        dictionary=loader.id2token,
        max_iter=500,
        convergence_threshold=1e-4,
        min_iter=50,
        doc_assignments=doc_assignments,
        hhi_scope=args.hhi_scope,
    )
 
    # Save top-10 topic words for LLM-as-a-Judge evaluation (LLMjudge.py)
    print("Extracting top-10 words per topic for LLM judge...")
    extracted_topics = lex_tm.get_top_topic_words(top_n=10)
 
    os.makedirs("experiment_logs1", exist_ok=True)
    topic_output_file = (
        f"experiment_logs1/{args.dataset_name}"
        f"_gamma_{args.gamma}_tau_{args.tau}_topics.json"
    )
    with open(topic_output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_topics, f, ensure_ascii=False, indent=4)
    print(f"Topics saved to {topic_output_file}")
 
    # -------------------------------------------------------
    # PHASE 3: RAG INDEXING & ROUTING
    # -------------------------------------------------------
    print("\n=== Phase 3: RAG Indexing & Routing ===")
 
    router = LexTMRouter(
        dense_model_name='BAAI/bge-m3',
        lex_tm_model=lex_tm,
        dictionary=loader,
        language=lang,
        min_query_tokens=args.min_query_tokens,
    )
 
    # cache_path encodes everything that affects embeddings and topic
    # distributions (dataset, tau, K) but NOT gamma — gamma only affects
    # the final interpolation in retrieve() so all gamma ablation runs
    # for the same (dataset, tau, K) share one cache, saving repeated
    # BGE-m3 encoding passes.
    cache_path = (
        f"experiment_logs1/cache_{args.dataset_name}"
        f"_tau{args.tau}_K{args.num_topics}"
    )
    router.index_corpus(all_chunks_dicts, cache_path=cache_path)
 
    # -------------------------------------------------------
    # PHASE 4: EXTRINSIC EVALUATION
    # -------------------------------------------------------
    print("\n=== Phase 4: Extrinsic Retrieval Evaluation ===")
 
    if args.dataset_name in _QRELS_DATASETS:
        # MIRACL and Mr. TyDi use human-annotated qrels
        if not args.queries_path or not args.qrels_path:
            raise ValueError(
                f"'{args.dataset_name}' requires --queries_path and --qrels_path. "
                "Run the corresponding download script first."
            )
        queries, qrels = load_queries_and_qrels(args.queries_path, args.qrels_path)
        id_prefix = _QRELS_DATASETS[args.dataset_name]
        eval_dataset = build_evaluation_dataset_qrels(
            queries, qrels, all_chunks_dicts, id_prefix
        )
    else:
        # All other datasets use the title-as-query proxy
        eval_dataset = build_evaluation_dataset(df, all_chunks_dicts, args.dataset_name)
 
    evaluate_retrieval(router, eval_dataset, gamma=args.gamma)
    #bootstrap_mrr_ci(router, eval_dataset, gamma=args.gamma)
 
    print(f"\nPipeline complete — {args.dataset_name} (gamma={args.gamma}, tau={args.tau}).")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lex-TM RAG Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        choices=[
            'english_aviation', 'chinese_aviation',
            'trec_covid', 'nfcorpus',
            'thucnews', 'sogou',
            'miracl_zh',
            'mr_tydi_ja', 'mr_tydi_th',
            'medweb_en', 'medweb_ja', 'medweb_zh',
        ],
        help="Dataset identifier — controls loader, language, and evaluation mode.",
    )
    parser.add_argument(
        "--filepath", type=str, required=True,
        help="Path to the dataset file. "
             "For medweb_en pass medweb_rag_en.csv; "
             "for medweb_ja pass medweb_rag_ja_fixed.csv; "
             "for medweb_zh pass medweb_rag_zh_fixed.csv. "
             "The merged medweb_rag_all_fixed.csv also works for any variant.",
    )
    parser.add_argument(
        "--queries_path", type=str, default=None,
        help="Path to queries.jsonl — required only for miracl_zh and mr_tydi_*.",
    )
    parser.add_argument(
        "--qrels_path", type=str, default=None,
        help="Path to qrels/dev.tsv — required only for miracl_zh and mr_tydi_*.",
    )
    parser.add_argument(
        "--max_docs", type=int, default=None,
        help="Cap the corpus at this many documents before chunking. "
             "Documents are sampled with random_state=42 for reproducibility. "
             "Disclose the cap value in the paper. Default: None (no cap).",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=150,
        help="Number of tokens per English chunk. Default 150. "
             "Use --chunk_size 20 for MedWeb (texts avg ~12 words; "
             "default would discard ~98%% of documents).",
    )
    parser.add_argument(
        "--chunk_size_zh", type=int, default=None,
        help="Tokens per Chinese/Japanese chunk (jieba/MeCab). "
             "Defaults to chunk_size * 2. Use --chunk_size_zh 30 for MedWeb.",
    )
    parser.add_argument(
        "--chunk_size_ja", type=int, default=None,
        help="Tokens per Japanese chunk (MeCab morphemes). "
             "Defaults to chunk_size * 2. Use --chunk_size_ja 30 for MedWeb.",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.7,
        help="Dense-topic interpolation weight γ ∈ [0, 1]. "
             "γ=1.0 → pure dense; γ=0.0 → pure Lex-TM topic routing.",
    )
    parser.add_argument(
        "--tau", type=float, default=0.5,
        help="Herfindahl exclusivity threshold τ. "
             "Tokens with Ev > τ get their prior amplified. "
             "τ > 1.0 disables all amplification (Standard LDA baseline).",
    )
    parser.add_argument(
        "--num_topics", type=int, default=50,
        help="Number of latent topics K for Gibbs sampling.",
    )
    parser.add_argument(
        "--lambda_amp", type=float, default=2.0,
        help="Prior amplification factor λ applied to exclusive tokens (Eq. 4).",
    )
    parser.add_argument(
    "--hhi_scope", type=str, default='document',
        choices=['document', 'chunk'],
        help="Granularity for HHI exclusivity computation. "
             "'document' aggregates all chunks of a source document (paper default). "
             "'chunk' treats each chunk independently (ablation).",
    )
    parser.add_argument(
        "--min_query_tokens", type=int, default=0,
        help="Minimum BoW token count for query topic routing. Queries with "
             "fewer tokens fall back to pure dense (effective_gamma=1.0). "
             "0 disables the gate (original behaviour). "
             "Recommended: 5 for short-query corpora (MIRACL, Mr. TyDi).",
    )
    parser.add_argument(
        "--run_bm25", action="store_true", default=False,
        help="Also evaluate BM25+Dense RRF baseline after main evaluation. "
             "Requires rank_bm25 (pip install rank-bm25). "
             "Use only on primary result datasets — not sanity checks.",
    )
 
    args = parser.parse_args()
    run_end_to_end_pipeline(args)