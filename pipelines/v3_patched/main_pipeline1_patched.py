"""
main_pipeline1_patched.py
==========================
Drop-in replacement for main_pipeline1.py.

Fixes vs original
-----------------
FIX A — --run_bm25 flag is now wired up. Original parsed the argument but
         never used it — HybridRouter was never imported or instantiated.
         Now, when --run_bm25 is passed, the pipeline:
           1. Instantiates HybridRouter (which builds the BM25 index at
              index_corpus time on top of the cached dense embeddings)
           2. Calls evaluate_retrieval_rrf() on the same eval_dataset
           3. Calls bootstrap_mrr_ci() on the RRF scores

FIX B — evaluate_retrieval now returns rr_scores; bootstrap_mrr_ci takes
         those scores directly (no retrieval re-run). Import list updated.

FIX C — Cache directory is experiment_logs1/ (matching the original script)
         so existing K=50 embeddings are found automatically.
         Validation pipeline uses experiment_logs/ — see copy_cache.sh to
         reuse embeddings across both pipelines without re-encoding.

Everything else (data loading, Lex-TM training, eval dataset construction)
is identical to the original main_pipeline1.py.

Usage
-----
  # Lex-TM only (same as before)
  python main_pipeline1_patched.py --dataset_name medweb_en --filepath ... --gamma 0.7 --tau 0.5 ...

  # Lex-TM + RRF baseline (NEW — triggers HybridRouter)
  python main_pipeline1_patched.py --dataset_name medweb_en --filepath ... --gamma 0.7 --tau 0.5 --run_bm25 ...
"""

import csv
import json
import os
import re
import numpy as np
import pandas as pd
import argparse
from rag_dataload import UniversalRAGLoader
from lex_tm_model import LexTMLdaModel

# FIX B: import HybridRouter and evaluate_retrieval_rrf from patched pipeline
from rag_evaluation_pipeline_patched import (
    LexTMRouter,
    HybridRouter,
    evaluate_retrieval,
    evaluate_retrieval_rrf,
    evaluate_retrieval_bm25,
    bootstrap_mrr_ci,
    paired_bootstrap_diff,
)
from typing import Dict, List, Tuple


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

_DATASET_LANG: Dict[str, str] = {
    'english_aviation': 'en',
    'chinese_aviation': 'zh',
    'trec_covid':       'en',
    'nfcorpus':         'en',
    'chatdoctor':       'en',
    'cmedqa':           'zh',
    'thucnews':         'zh',
    'sogou':            'zh',
    'miracl_zh':        'zh',
    'mr_tydi_ja':       'ja',
    'mr_tydi_th':       'th',
    'medweb_en':        'en',
    'medweb_ja':        'ja',
    'medweb_zh':        'zh',
}

_QRELS_DATASETS: Dict[str, str] = {
    'miracl_zh':  'miracl_',
    'mr_tydi_ja': 'mr_tydi_ja_',
    'mr_tydi_th': 'mr_tydi_th_',
    'trec_covid': 'trec_covid_',
    'nfcorpus':   'nfcorpus_',
    # Added: short-form medical QA retrieval (real Q&A relevance, not
    # title/symptom-set matching). Second medical benchmark.
    # avg query 425 chars / avg doc 605 chars — shorter than
    # NFCorpus/TREC-COVID, though longer than MedWeb's ~12-token queries.
    'chatdoctor': 'chatdoctor_',
    # Added: Chinese medical Q&A retrieval (real physician-answer relevance).
    # Stratified, disclosed corpus subsample — see export_cmedqa_to_beir.py
    # for the sampling methodology and required disclosure text.
    'cmedqa': 'cmedqa_',
}


def build_evaluation_dataset(df: pd.DataFrame, chunks: list, dataset_type: str) -> list:
    """Unchanged from main_pipeline1.py."""
    print(f"Building evaluation pairs for {dataset_type}...")

    query_col = QUERY_COLUMN.get(dataset_type)
    if query_col is None:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'.")

    chunk_lookup: Dict[str, set] = {}
    for chunk in chunks:
        doc_id = chunk['doc_id']
        if doc_id not in chunk_lookup:
            chunk_lookup[doc_id] = set()
        chunk_lookup[doc_id].add(chunk['chunk_id'])

    title_groups: Dict[str, Dict] = {}
    skipped = 0

    for _, row in df.iterrows():
        doc_id = row['document_id']
        query  = str(row.get(query_col, '')).strip()

        if not query or doc_id not in chunk_lookup:
            skipped += 1
            continue

        if query not in title_groups:
            title_groups[query] = {'chunk_ids': set(), 'rep_doc_id': doc_id}
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
    print(f"Generated {n_groups} evaluation pairs from {n_docs} documents "
          f"({skipped} skipped). Avg docs per title group: {avg_size:.1f}")
    if avg_size > 2.0:
        print(f"  NOTE: avg group size {avg_size:.1f} > 1 — title grouping active.")
    return eval_data


def load_queries_and_qrels(queries_path: str, qrels_path: str
                           ) -> Tuple[Dict[str, str], List[Dict]]:
    """Unchanged from main_pipeline1.py."""
    queries: Dict[str, str] = {}
    with open(queries_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            queries[str(record['_id'])] = record['text']

    qrels: List[Dict] = []
    with open(qrels_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(
            f, fieldnames=['query_id', 'corpus_id', 'score'], delimiter='\t'
        )
        for row in reader:
            try:
                score_val = int(row['score'])
            except ValueError:
                continue
            if score_val > 0:
                qrels.append({
                    'query_id':  str(row['query_id']),
                    'corpus_id': str(row['corpus_id']),
                    'score':     score_val,
                })

    print(f"Loaded {len(queries):,} queries and {len(qrels):,} qrel pairs.")
    return queries, qrels


def build_evaluation_dataset_qrels(queries, qrels, all_chunks_dicts, id_prefix):
    """Unchanged from main_pipeline1.py."""
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

    print(f"Generated {len(eval_data):,} evaluation pairs "
          f"({skipped_no_query} skipped — query not found, "
          f"{skipped_no_chunks} skipped — document not in corpus).")
    return eval_data


def run_end_to_end_pipeline(args):
    print(f"\n{'='*50}")
    print(f"Lex-TM Pipeline — {args.dataset_name.upper()}")
    print(f"  gamma={args.gamma} | tau={args.tau} | K={args.num_topics} | "
          f"lambda={args.lambda_amp} | run_bm25={args.run_bm25}")
    print(f"  encoder={args.dense_model} | query_prompt={args.query_prompt_name}")
    print(f"{'='*50}")

    # -------------------------------------------------------
    # PHASE 1: DATA LOADING
    # -------------------------------------------------------
    print("\n=== Phase 1: Data Loading & Preprocessing ===")

    lang = _DATASET_LANG.get(args.dataset_name, 'en')

    loader = UniversalRAGLoader(
        chunk_size    = args.chunk_size,
        chunk_size_zh = args.chunk_size_zh if args.chunk_size_zh else args.chunk_size * 2,
        chunk_size_ja = args.chunk_size_ja if args.chunk_size_ja else args.chunk_size * 2,
        overlap       = max(1, args.chunk_size // 7),
        min_df        = 2,
    )

    loaders = {
        'english_aviation': loader.load_english_aviation,
        'chinese_aviation': loader.load_chinese_aviation,
        'thucnews':         loader.load_thucnews,
        'sogou':            loader.load_sogou,
        'miracl_zh':        loader.load_miracl_zh,
        # BUGFIX (#18): the original had 'trec_covid': loader.load_beir and
        # 'nfcorpus': loader.load_beir as BARE entries (defaulting to
        # prefix="nfcorpus_"), immediately followed by duplicate-key
        # corrected lambda entries with explicit prefixes. Python's
        # last-key-wins dict semantics meant the original silently worked
        # despite looking redundant — but an earlier version of this
        # file (this patched pipeline) kept only the first, bare, buggy
        # entry and dropped the correcting duplicate. That meant TREC-COVID
        # documents got prefixed 'nfcorpus_' instead of 'trec_covid_',
        # breaking the _QRELS_DATASETS lookup and silently producing ZERO
        # valid evaluation pairs. Fixed here with explicit prefixes and no
        # duplicate keys, matching the ORIGINAL's actual runtime behaviour.
        'trec_covid':       lambda fp: loader.load_beir(fp, prefix='trec_covid_'),
        'nfcorpus':         lambda fp: loader.load_beir(fp, prefix='nfcorpus_'),
        # Added: second medical benchmark (real Q&A relevance judgments,
        # not title/symptom-set matching) — see docs/FIXES.md.
        'chatdoctor':       lambda fp: loader.load_beir(fp, prefix='chatdoctor_'),
        'cmedqa':           lambda fp: loader.load_beir(fp, prefix='cmedqa_'),
        'mr_tydi_ja':       lambda fp: loader.load_mr_tydi(fp, 'ja'),
        'mr_tydi_th':       lambda fp: loader.load_mr_tydi(fp, 'th'),
        'medweb_en':        lambda fp: loader.load_medweb(fp, 'en'),
        'medweb_ja':        lambda fp: loader.load_medweb(fp, 'ja'),
        'medweb_zh':        lambda fp: loader.load_medweb(fp, 'zh'),
    }
    df = loaders[args.dataset_name](args.filepath)

    if df.empty:
        raise RuntimeError(f"DataFrame is empty after loading '{args.filepath}'.")

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
        num_topics  = args.num_topics,
        alpha       = 0.1,
        beta_base   = 0.01,
        tau         = args.tau,
        lambda_amp  = args.lambda_amp,
        random_state= 42,
    )
    lex_tm.fit(
        corpus               = corpus_bow,
        dictionary           = loader.id2token,
        max_iter             = 500,
        convergence_threshold= 1e-4,
        min_iter             = 50,
        doc_assignments      = doc_assignments,
        hhi_scope            = args.hhi_scope,
    )

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
    # FIX A: use HybridRouter when --run_bm25 is set so BM25 index is
    # built at index_corpus time (reuses cached dense embeddings).
    # -------------------------------------------------------
    print("\n=== Phase 3: RAG Indexing & Routing ===")

    # -------------------------------------------------------------------
    # CACHE KEYING — CRITICAL when swapping dense encoders.
    #
    # The embedding cache MUST include the model name. Otherwise a Qwen run
    # would silently load BGE-m3 vectors from experiment_logs1/, invalidating
    # the entire experiment. We sanitise the model id (e.g. 'Qwen/Qwen3-
    # Embedding-0.6B' -> 'Qwen_Qwen3-Embedding-0.6B') into the cache prefix.
    #
    # Topic distributions are model-INDEPENDENT (Gibbs sampling on BoW, never
    # sees embeddings), so they are keyed only by (dataset, tau, K) and are
    # shared across encoders. index_corpus resolves the two files separately.
    # -------------------------------------------------------------------
    model_tag = re.sub(r'[^A-Za-z0-9._-]', '_', args.dense_model)

    emb_cache_path = (
        f"experiment_logs1/emb_{args.dataset_name}"
        f"_{model_tag}_chunk{args.chunk_size}"
    )
    tdist_cache_path = (
        f"experiment_logs1/tdist_{args.dataset_name}"
        f"_tau{args.tau}_K{args.num_topics}"
    )

    # For BGE-m3 no query prompt; for Qwen3-Embedding use its 'query' prompt.
    router_kwargs = dict(
        dense_model_name  = args.dense_model,
        lex_tm_model      = lex_tm,
        dictionary        = loader,
        language          = lang,
        min_query_tokens  = args.min_query_tokens,
        # FIX 3 & 4: controlled via CLI so individual ablation runs can
        # disable either improvement. Defaults active for MedWeb.
        query_smoothing   = args.query_smoothing,
        adaptive_gamma    = args.adaptive_gamma,
        query_prompt_name = args.query_prompt_name,
        embed_batch_size  = args.embed_batch_size,
        model_dtype       = args.model_dtype,
        trust_remote_code = args.trust_remote_code,
    )

    if args.run_bm25:
        # HybridRouter is a subclass of LexTMRouter — all retrieve() calls
        # are identical. The BM25 index is built on top at index time.
        router = HybridRouter(**router_kwargs)
    else:
        router = LexTMRouter(**router_kwargs)

    router.index_corpus(
        all_chunks_dicts,
        emb_cache_path=emb_cache_path,
        tdist_cache_path=tdist_cache_path,
    )

    # -------------------------------------------------------
    # PHASE 4: EXTRINSIC EVALUATION
    # -------------------------------------------------------
    print("\n=== Phase 4: Extrinsic Retrieval Evaluation ===")

    if args.dataset_name in _QRELS_DATASETS:
        if not args.queries_path or not args.qrels_path:
            raise ValueError(
                f"'{args.dataset_name}' requires --queries_path and --qrels_path."
            )
        queries, qrels = load_queries_and_qrels(args.queries_path, args.qrels_path)
        id_prefix    = _QRELS_DATASETS[args.dataset_name]
        eval_dataset = build_evaluation_dataset_qrels(
            queries, qrels, all_chunks_dicts, id_prefix
        )
    else:
        eval_dataset = build_evaluation_dataset(df, all_chunks_dicts, args.dataset_name)

    # Lex-TM evaluation
    # FIX B: evaluate_retrieval now returns rr_scores for bootstrap
    rr_scores = evaluate_retrieval(router, eval_dataset, gamma=args.gamma)

    print("\n--- Bootstrap 95% CI (Lex-TM) ---")
    # FIX B: pass pre-computed scores — no retrieval re-run
    bootstrap_mrr_ci(rr_scores)

    # FIX A: RRF baseline — only runs when --run_bm25 is passed
    if args.run_bm25:
        print("\n=== Pure BM25 Baseline ===")
        # Isolates the lexical baseline: answers "is Lex-TM's gain just
        # lexical matching?" Same corpus/tokenizer/eval as Lex-TM.
        rr_bm25 = evaluate_retrieval_bm25(router, eval_dataset)
        print("\n--- Bootstrap 95% CI (Pure BM25) ---")
        bootstrap_mrr_ci(rr_bm25)

        print("\n=== BM25+Dense RRF Baseline ===")
        rr_rrf = evaluate_retrieval_rrf(router, eval_dataset, k=60)
        print("\n--- Bootstrap 95% CI (RRF) ---")
        bootstrap_mrr_ci(rr_rrf)

        # Honest paired significance test: does Lex-TM actually beat RRF /
        # Pure BM25, or do the point estimates just happen to differ?
        # Independent CIs (above) are conservative and can't establish this —
        # overlapping CIs mean "can't distinguish," not "parity," and
        # non-overlapping CIs mean "probably differs," not "proven." Paired
        # bootstrap on the SAME resampled queries for both arms is the
        # correct test. This will not manufacture a win: if Lex-TM's scores
        # are not genuinely higher, the reported CI on the difference will
        # include zero, and the verdict will say so plainly.
        print("\n=== Paired significance: Lex-TM vs. baselines ===")
        paired_bootstrap_diff(rr_scores, rr_rrf, label_a="Lex-TM", label_b="RRF")
        paired_bootstrap_diff(rr_scores, rr_bm25, label_a="Lex-TM", label_b="Pure BM25")

    print(f"\nPipeline complete — {args.dataset_name} "
          f"(gamma={args.gamma}, tau={args.tau}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lex-TM RAG Evaluation Pipeline (patched)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", type=str, required=True,
        choices=['english_aviation','chinese_aviation','trec_covid','nfcorpus',
                 'chatdoctor','cmedqa','thucnews','sogou','miracl_zh','mr_tydi_ja',
                 'mr_tydi_th','medweb_en','medweb_ja','medweb_zh'])
    parser.add_argument("--filepath",       type=str, required=True)
    parser.add_argument("--queries_path",   type=str, default=None)
    parser.add_argument("--qrels_path",     type=str, default=None)
    parser.add_argument("--max_docs",       type=int, default=None)
    parser.add_argument("--chunk_size",     type=int, default=150)
    parser.add_argument("--chunk_size_zh",  type=int, default=None)
    parser.add_argument("--chunk_size_ja",  type=int, default=None)
    parser.add_argument("--gamma",          type=float, default=0.7)
    parser.add_argument("--tau",            type=float, default=0.5)
    parser.add_argument("--num_topics",     type=int, default=50)
    parser.add_argument("--lambda_amp",     type=float, default=2.0)
    parser.add_argument("--hhi_scope",      type=str, default='document',
                        choices=['document', 'chunk'])
    parser.add_argument("--dense_model", type=str, default='BAAI/bge-m3',
        help="HuggingFace/sentence-transformers dense encoder id. "
             "e.g. 'BAAI/bge-m3' (default) or 'Qwen/Qwen3-Embedding-0.6B'.")
    parser.add_argument("--query_prompt_name", type=str, default=None,
        help="sentence-transformers registered prompt name for QUERIES. "
             "Leave None for BGE-m3. Use 'query' for Qwen3-Embedding "
             "(instruction models drop 1-5%% MRR without it). Documents are "
             "never prompted.")
    parser.add_argument("--embed_batch_size", type=int, default=256,
        help="Encode batch size. Lower for large models (e.g. 32-64 for 8B).")
    parser.add_argument("--model_dtype", type=str, default=None,
        help="torch dtype for the encoder, e.g. 'float16' for 8B models to "
             "fit VRAM. None uses the model default.")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
        help="Pass trust_remote_code=True to SentenceTransformer. REQUIRED "
             "for some models (e.g. Snowflake/snowflake-arctic-embed-*-v2.0) "
             "which ship custom modeling code -- check the model card before "
             "enabling. Off by default since it executes arbitrary repo code.")
    parser.add_argument("--min_query_tokens", type=int, default=0)
    parser.add_argument("--query_smoothing", type=float, default=0.1,
        help="FIX 3: Blend query topic distribution toward uniform prior. "
             "0.0 disables (original behaviour). The default 0.1 is kept for "
             "backward compatibility; it lowers MRR, and every run in the paper "
             "passes 0.0 (docs/FIXES.md #10).")
    parser.add_argument("--adaptive_gamma", action="store_true", default=True,
        help="FIX 4: Reduce gamma by up to 0.2 when query topic distribution "
             "is concentrated. Pass --no_adaptive_gamma to disable.")
    parser.add_argument("--no_adaptive_gamma", dest="adaptive_gamma",
                        action="store_false")
    parser.add_argument("--run_bm25", action="store_true", default=False,
        help="FIX A: Now wired up. Builds BM25 index via HybridRouter and "
             "reports RRF (k=60) results alongside Lex-TM.")

    args = parser.parse_args()
    run_end_to_end_pipeline(args)
