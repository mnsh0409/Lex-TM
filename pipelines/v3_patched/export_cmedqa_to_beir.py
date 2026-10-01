"""
export_cmedqa_to_beir.py
=========================
Exports mteb/CmedqaRetrieval (Chinese medical Q&A retrieval, real patient
question / physician answer pairs, CMedQAv2-derived) to local BEIR-format
files.

HONESTY NOTE — read before citing anything from this dataset
--------------------------------------------------------------
The official corpus has 100,001 documents. Encoding the full corpus with
BGE-m3 is not needed for the relative comparison, so this script writes a
STRATIFIED SUBSAMPLE, not the full official benchmark:

    1. Every document referenced in ANY qrel is kept (so no query is ever
       silently orphaned by the subsample — every query remains answerable).
    2. The corpus is filled with a random sample of remaining documents up
       to --target_corpus_size (default 10,000).
    3. ALL 3,999 queries are kept (only the corpus is capped; encoding is
       the bottleneck, not evaluation).

This means: (a) results from this export are NOT comparable to the official
CmedqaRetrieval leaderboard, which uses the full 100K corpus, and that must
be disclosed alongside any number quoted from this run; (b) the RELATIVE
comparison between Lex-TM / BM25 / Dense / RRF on this subsample remains
valid, since every method sees the identical corpus.

Known caveat inherited from the source dataset (documented in prior
benchmarking literature, e.g. "Benchmarking and Enabling Efficient Chinese
Medical Retrieval via Asymmetric Encoders"): CmedqaRetrieval's relevance
judgments are built from labeled Q&A pairs without exhaustively annotating
all other potentially-relevant candidates, so some true positives may be
scored as irrelevant (false negatives in the qrels). This affects the
official benchmark too, not just this subsample.

Schema confirmed by direct inspection (2026-07-12):
    corpus config : {'_id': ..., 'title': ..., 'text': ...}  (100,001 rows)
    queries config: {'_id': ..., 'text': ...}                (3,999 rows)
    default config (qrels): {'query-id': ..., 'corpus-id': ..., 'score': 1}
                    NOT 1:1 — 1 to 19 relevant docs per query, avg 1.86.

Usage
-----
    python export_cmedqa_to_beir.py --out_dir ../data/beir/cmedqa \
        --target_corpus_size 10000

Then run with the SAME locked config used for ChatDoctorRetrieval (no new
hyperparameter tuning for this dataset):
    python main_pipeline1_patched.py --dataset_name cmedqa \
        --filepath     ../data/beir/cmedqa/corpus.jsonl \
        --queries_path ../data/beir/cmedqa/queries.jsonl \
        --qrels_path   ../data/beir/cmedqa/qrels/test.tsv \
        --gamma 0.7 --tau 0.8 --num_topics 50 \
        --query_smoothing 0.0 --no_adaptive_gamma --run_bm25
    (chunk_size defaults to 150, matching the ChatDoctor run — not re-tuned.)
"""

import argparse
import json
import os
import random


def main(args):
    from datasets import load_dataset  # deferred import — only needed here

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "qrels"), exist_ok=True)
    rng = random.Random(42)  # fixed seed for the subsample — reproducible

    # ---- Load all three configs (each downloaded exactly ONCE) -----------
    print("Downloading corpus config (100,001 docs, ~few min)...")
    corpus_dd = load_dataset("mteb/CmedqaRetrieval", "corpus")
    corpus = corpus_dd[list(corpus_dd.keys())[0]]  # single-split DatasetDict either way
    print(f"  Corpus split resolved as: '{list(corpus_dd.keys())[0]}'")

    print("Downloading queries config...")
    queries_dd = load_dataset("mteb/CmedqaRetrieval", "queries")
    queries = queries_dd[list(queries_dd.keys())[0]]
    print(f"  Queries split resolved as: '{list(queries_dd.keys())[0]}'")

    print("Downloading qrels (default config)...")
    qrels_dd = load_dataset("mteb/CmedqaRetrieval", "default")
    qrels_raw = qrels_dd[list(qrels_dd.keys())[0]]
    print(f"  Qrels split resolved as: '{list(qrels_dd.keys())[0]}'")

    print(f"  Full corpus: {len(corpus):,} | queries: {len(queries):,} "
          f"| qrel pairs: {len(qrels_raw):,}")

    # ---- Detect id-field convention (DON'T assume — ChatDoctorRetrieval
    # used 'id', not '_id'; guessing wrong here would silently break the
    # join downstream, same risk we caught before building the ChatDoctor
    # export) --------------------------------------------------------------
    corpus_id_field = "_id" if "_id" in corpus.column_names else "id"
    queries_id_field = "_id" if "_id" in queries.column_names else "id"
    print(f"  Corpus id field detected as: '{corpus_id_field}' "
          f"(columns: {corpus.column_names})")
    print(f"  Queries id field detected as: '{queries_id_field}' "
          f"(columns: {queries.column_names})")
    print(f"  Qrels columns: {qrels_raw.column_names}")

    # ---- Stratified corpus subsample --------------------------------------
    gold_ids = {row["corpus-id"] for row in qrels_raw}
    print(f"\nStratified subsampling (seed=42, target={args.target_corpus_size:,}):")
    print(f"  Unique gold (referenced-in-qrels) documents: {len(gold_ids):,}")

    corpus_by_id = {row[corpus_id_field]: row for row in corpus}
    gold_docs = [corpus_by_id[i] for i in gold_ids if i in corpus_by_id]
    missing_gold = len(gold_ids) - len(gold_docs)
    if missing_gold:
        print(f"  WARNING: {missing_gold} gold ids not found in corpus config — "
              f"those qrel rows will be unresolvable regardless of sampling.")

    remaining_pool = [row for row in corpus if row[corpus_id_field] not in gold_ids]
    fill_n = max(0, args.target_corpus_size - len(gold_docs))
    fill_docs = rng.sample(remaining_pool, min(fill_n, len(remaining_pool)))
    final_corpus = gold_docs + fill_docs
    rng.shuffle(final_corpus)

    print(f"  Final subsample: {len(gold_docs):,} gold + {len(fill_docs):,} "
          f"random fill = {len(final_corpus):,} total documents")
    print(f"  (Official benchmark uses the full {len(corpus):,}-doc corpus — "
          f"this is a disclosed, gold-preserving subsample, not the full set.)")

    # ---- Write corpus.jsonl ------------------------------------------------
    corpus_path = os.path.join(args.out_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in final_corpus:
            f.write(json.dumps({
                "_id":   row[corpus_id_field],
                "title": row.get("title", "") or "",
                "text":  row["text"],
            }, ensure_ascii=False) + "\n")
    print(f"\n  Wrote {len(final_corpus):,} documents -> {corpus_path}")

    # ---- Write queries.jsonl (ALL queries kept) ----------------------------
    queries_path = os.path.join(args.out_dir, "queries.jsonl")
    with open(queries_path, "w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps({
                "_id":  row[queries_id_field],
                "text": row["text"],
            }, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(queries):,} queries -> {queries_path}")

    # ---- Write qrels/test.tsv (multi-relevant-doc: group by query) --------
    final_corpus_ids = {row[corpus_id_field] for row in final_corpus}
    qrels_path = os.path.join(args.out_dir, "qrels", "test.tsv")
    kept, dropped = 0, 0
    with open(qrels_path, "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in qrels_raw:
            if row["corpus-id"] in final_corpus_ids:
                f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
                kept += 1
            else:
                dropped += 1  # only possible if missing_gold > 0 above
    print(f"  Wrote {kept:,} qrel pairs -> {qrels_path}"
          + (f"  ({dropped} dropped — see WARNING above)" if dropped else ""))

    # ---- Sanity check: every query has at least one answerable qrel -------
    print("\nSanity check: does every query retain at least one resolvable qrel?")
    query_ids = {row[queries_id_field] for row in queries}
    answerable_queries = {row["query-id"] for row in qrels_raw if row["corpus-id"] in final_corpus_ids}
    orphaned = query_ids - answerable_queries
    # queries with NO qrel at all in the source (not a sampling artifact)
    queries_with_any_qrel = {row["query-id"] for row in qrels_raw}
    orphaned_by_sampling = orphaned & queries_with_any_qrel
    print(f"  Queries with zero resolvable relevant doc after subsampling: "
          f"{len(orphaned_by_sampling)} / {len(queries_with_any_qrel)}")
    if orphaned_by_sampling:
        print("  WARNING: non-zero — the gold-preservation guarantee failed for "
              "these; investigate before running the pipeline.")
    else:
        print("  All queries with a qrel in the source remain answerable. "
              "Gold-preservation guarantee held.")

    print(f"\nDone. Point --filepath/--queries_path/--qrels_path at:")
    print(f"  {corpus_path}")
    print(f"  {queries_path}")
    print(f"  {qrels_path}")
    print(f"\nDISCLOSURE TEXT (paste into any writeup that cites this run):")
    print(f'  "Evaluated on a stratified {len(final_corpus):,}-document subsample of '
          f'CmedqaRetrieval (official corpus: {len(corpus):,} docs), constructed to '
          f'preserve every gold-relevant document plus a random fill to the target '
          f'size (seed=42); all {len(queries):,} queries retained. Not directly '
          f'comparable to the official full-corpus leaderboard."')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../data/beir/cmedqa",
        help="Output directory for corpus.jsonl, queries.jsonl, qrels/test.tsv")
    parser.add_argument("--target_corpus_size", type=int, default=10000,
        help="Target corpus size after stratified subsampling (gold docs always kept)")
    main(parser.parse_args())
