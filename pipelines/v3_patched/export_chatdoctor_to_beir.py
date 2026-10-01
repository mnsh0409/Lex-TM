"""
export_chatdoctor_to_beir.py
=============================
Exports mteb/ChatDoctorRetrieval to local BEIR-format files, matching the
exact schema `load_beir()` and `load_queries_and_qrels()` already expect
(see rag_dataload.py / main_pipeline1_patched.py). Zero new parsing code is
needed in the pipeline itself — this dataset reuses load_beir() unchanged,
exactly like TREC-COVID and NFCorpus already do.

Dataset: real ChatDoctor-HealthCareMagic Q&A pairs (patient question -> real
doctor response). Relevance is 1:1 (max_relevant_docs_per_query == 1), drawn
from genuine question/answer correspondence, not title or symptom-set
matching. Corpus avg 605 chars, queries avg 425 chars — shorter than
TREC-COVID/NFCorpus, though longer than MedWeb's ~12-token queries.

Schema confirmed by direct inspection (2026-07-09):
    corpus  : {'id': '000000000', 'text': ..., 'title': ''}   (9-digit, zero-padded)
    queries : {'id': '00000',     'text': ...}                (5-digit, zero-padded)
    qrels   : {'query-id': '00000', 'corpus-id': '000000000', 'score': 1}
Corpus 'id' format was verified to match qrels 'corpus-id' format exactly
before writing this script — a mismatch here would silently produce zero
evaluation pairs (build_evaluation_dataset_qrels skips unmatched doc_ids
without raising an error).

Usage
-----
    python export_chatdoctor_to_beir.py --out_dir ../data/beir/chatdoctor

Then run exactly like TREC-COVID / NFCorpus:
    python main_pipeline1_patched.py --dataset_name chatdoctor \
        --filepath   ../data/beir/chatdoctor/corpus.jsonl \
        --queries_path ../data/beir/chatdoctor/queries.jsonl \
        --qrels_path   ../data/beir/chatdoctor/qrels/test.tsv \
        --gamma 0.7 --tau 0.5 --num_topics 50 \
        --query_smoothing 0.0 --no_adaptive_gamma --run_bm25
"""

import argparse
import json
import os


def main(args):
    from datasets import load_dataset  # deferred import — only needed here

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "qrels"), exist_ok=True)

    # ---- Corpus -------------------------------------------------------
    print("Downloading corpus config...")
    corpus = load_dataset("mteb/ChatDoctorRetrieval", "corpus")["test"]
    corpus_path = os.path.join(args.out_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for row in corpus:
            # BEIR schema: _id / title / text (matches load_beir()'s
            # doc["_id"], doc.get('title',''), doc['text'] exactly).
            f.write(json.dumps({
                "_id":   row["id"],
                "title": row.get("title", "") or "",
                "text":  row["text"],
            }, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(corpus):,} documents -> {corpus_path}")

    # ---- Queries --------------------------------------------------------
    print("Downloading queries config...")
    queries = load_dataset("mteb/ChatDoctorRetrieval", "queries")["test"]
    queries_path = os.path.join(args.out_dir, "queries.jsonl")
    with open(queries_path, "w", encoding="utf-8") as f:
        for row in queries:
            # load_queries_and_qrels() expects record['_id'] / record['text']
            f.write(json.dumps({
                "_id":  row["id"],
                "text": row["text"],
            }, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(queries):,} queries -> {queries_path}")

    # ---- Qrels ----------------------------------------------------------
    print("Downloading qrels config...")
    qrels = load_dataset("mteb/ChatDoctorRetrieval", "qrels")["test"]
    qrels_path = os.path.join(args.out_dir, "qrels", "test.tsv")
    with open(qrels_path, "w", encoding="utf-8") as f:
        # load_queries_and_qrels() force-assigns column names positionally
        # (query_id, corpus_id, score) via csv.DictReader(fieldnames=...),
        # so a header row's exact spelling doesn't matter — it gets safely
        # skipped when int("query-id") fails. Written here for readability.
        f.write("query-id\tcorpus-id\tscore\n")
        for row in qrels:
            f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")
    print(f"  Wrote {len(qrels):,} qrel pairs -> {qrels_path}")

    # ---- Sanity check: every qrel resolves to a real corpus doc --------
    print("\nSanity check: qrels <-> corpus id join...")
    corpus_ids = {row["id"] for row in corpus}
    query_ids  = {row["id"] for row in queries}
    missing_corpus = sum(1 for row in qrels if row["corpus-id"] not in corpus_ids)
    missing_query  = sum(1 for row in qrels if row["query-id"]  not in query_ids)
    print(f"  Qrels referencing a missing corpus id: {missing_corpus} / {len(qrels)}")
    print(f"  Qrels referencing a missing query id:  {missing_query} / {len(qrels)}")
    if missing_corpus or missing_query:
        print("  WARNING: non-zero mismatches above — investigate before running "
              "the pipeline, or evaluation will silently skip those pairs.")
    else:
        print("  All qrels resolve cleanly. Safe to run the pipeline.")

    print(f"\nDone. Point --filepath/--queries_path/--qrels_path at:")
    print(f"  {corpus_path}")
    print(f"  {queries_path}")
    print(f"  {qrels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../data/beir/chatdoctor",
        help="Output directory for corpus.jsonl, queries.jsonl, qrels/test.tsv")
    main(parser.parse_args())
