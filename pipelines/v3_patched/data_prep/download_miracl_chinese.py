"""
download_miracl_chinese.py
==========================
Downloads the MIRACL Chinese dataset from HuggingFace and saves it in a
format directly compatible with the Lex-TM pipeline loader.

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages)
is a multilingual IR benchmark. The Chinese subset uses Chinese Wikipedia
passages with human-annotated relevance judgements.

HuggingFace sources
-------------------
Queries + qrels : miracl/miracl        (name='zh')
Corpus          : miracl/miracl-corpus (name='zh')

License: Apache 2.0

Corpus size note
----------------
The full MIRACL Chinese corpus is ~4.9M Wikipedia passages. Training the
Lex-TM topic model on the full corpus would reproduce the THUCNews runtime
problem. This script caps the corpus at --max_corpus_size passages
(default 100,000) while guaranteeing every passage referenced in the qrels
is included (and the hard negatives of every query). The earlier-pipeline
MIRACL-zh results use this 100K-passage sample; the cap keeps Gibbs sampling
tractable.

Output structure
----------------
data/SOTA/miracl_zh/
    corpus.jsonl         {_id, title, text}  -- capped corpus
    queries.jsonl        {_id, text}
    qrels/dev.tsv        query_id, corpus_id, score (tab-separated)
    qrels/train.tsv      same (if requested)

Requirements
------------
    pip install datasets huggingface_hub

Usage
-----
    python download_miracl_chinese.py
    python download_miracl_chinese.py --output_dir ../data/SOTA/miracl_zh
    python download_miracl_chinese.py --max_corpus_size 50000
    python download_miracl_chinese.py --splits train dev
"""

import argparse
import json
import csv
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download MIRACL Chinese dataset from HuggingFace."
    )
    parser.add_argument(
        "--output_dir", type=str, default="../data/SOTA/miracl_zh",
        help="Directory to save the dataset. Default: ../data/SOTA/miracl_zh",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["dev"],
        choices=["train", "dev"],
        help="Query/qrel splits to download. Default: dev. "
             "Note: MIRACL test split has no public qrels.",
    )
    parser.add_argument(
        "--max_corpus_size", type=int, default=100_000,
        help="Maximum number of corpus passages to retain. All passages "
             "referenced in qrels are always kept regardless of this cap. "
             "Default: 100000. Set to -1 for the full 4.9M corpus (slow).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for corpus sampling. Default: 42.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HuggingFace cache directory. Defaults to ~/.cache/huggingface.",
    )
    return parser.parse_args()


def save_jsonl(records: list, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records  ->  {filepath}")


def save_tsv(rows: list, filepath: Path, fieldnames: list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        for row in rows:
            writer.writerow(row)
    print(f"  Saved {len(rows):,} rows      ->  {filepath}")


def collect_qrel_doc_ids(splits: list, cache_dir: str) -> tuple:
    """
    First pass: load queries and qrels, collect the set of corpus doc IDs
    that are referenced in relevance judgements. These must always be
    included in the corpus regardless of the size cap.

    Returns
    -------
    all_queries     : dict {query_id: query_text}
    qrel_rows       : dict {split: list of {query_id, corpus_id, score}}
    required_doc_ids: set of corpus doc IDs referenced in any qrel
    """
    from datasets import load_dataset

    print(f"\n[1/3] Loading queries and qrels for splits: {splits}")

    all_queries = {}
    qrel_rows = {split: [] for split in splits}
    required_doc_ids = set()

    for split in splits:
        print(f"  Fetching split: {split} ...")
        dataset = load_dataset(
            "miracl/miracl",
            "zh",
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print(f"  Queries in '{split}': {len(dataset):,}")

        skipped = 0
        for row in dataset:
            query_id   = str(row["query_id"])
            query_text = row["query"]
            all_queries[query_id] = query_text

            positives = row.get("positive_passages", [])
            if not positives:
                skipped += 1
                continue

            for passage in positives:
                doc_id = str(passage["docid"])
                required_doc_ids.add(doc_id)
                qrel_rows[split].append({
                    "query_id":  query_id,
                    "corpus_id": doc_id,
                    "score":     1,
                })

            # Include hard negatives in the corpus so retrieval is non-trivial
            for passage in row.get("negative_passages", []):
                required_doc_ids.add(str(passage["docid"]))

        if skipped:
            print(f"  WARNING: {skipped} queries had no positive passages.")

    print(f"\n  Total unique queries : {len(all_queries):,}")
    print(f"  Total qrel pairs     : "
          f"{sum(len(v) for v in qrel_rows.values()):,}")
    print(f"  Corpus IDs in qrels  : {len(required_doc_ids):,}")

    return all_queries, qrel_rows, required_doc_ids


def download_and_cap_corpus(
    output_dir: Path,
    required_doc_ids: set,
    max_corpus_size: int,
    seed: int,
    cache_dir: str,
) -> list:
    """
    Second pass: stream the MIRACL corpus, always keep qrel-referenced
    passages, then fill remaining slots with a random sample up to
    max_corpus_size.

    Streaming avoids loading the full 4.9M passages into RAM.
    Early exit once enough background passages have been collected.
    """
    from datasets import load_dataset

    print(f"\n[2/3] Downloading corpus (miracl/miracl-corpus, zh)...")

    if max_corpus_size == -1:
        print("  max_corpus_size=-1: downloading full corpus (~4.9M passages).")
        print("  WARNING: This will make Gibbs sampling very slow.")
    else:
        print(f"  Corpus cap        : {max_corpus_size:,} passages")
        print(f"  Guaranteed        : {len(required_doc_ids):,} qrel-referenced passages")

    corpus_stream = load_dataset(
        "miracl/miracl-corpus",
        "zh",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    required_passages = {}   # doc_id -> record  (always kept)
    background_pool   = []   # non-qrel passages for random sampling

    print("  Streaming corpus passages (this may take a few minutes)...")
    processed = 0

    for row in corpus_stream:
        doc_id = str(row["docid"])
        record = {
            "_id":   doc_id,
            "title": row.get("title", ""),
            "text":  row["text"],
        }

        if doc_id in required_doc_ids:
            required_passages[doc_id] = record
        elif max_corpus_size == -1:
            background_pool.append(record)
        else:
            # Collect up to 5x the remaining background slots needed,
            # then sample down — avoids storing all 4.9M in memory.
            remaining_slots = max_corpus_size - len(required_doc_ids)
            if len(background_pool) < remaining_slots * 5:
                background_pool.append(record)

        processed += 1
        if processed % 500_000 == 0:
            print(f"  Processed {processed:,} passages "
                  f"(required found: {len(required_passages):,})...")

        # Early exit once we have enough of both
        if (max_corpus_size != -1
                and len(required_passages) == len(required_doc_ids)
                and len(background_pool) >= (
                    max_corpus_size - len(required_doc_ids)) * 3):
            print(f"  Early exit at {processed:,} passages — "
                  "sufficient background pool collected.")
            break

    print(f"\n  Required passages found : {len(required_passages):,} "
          f"/ {len(required_doc_ids):,}")

    missing = required_doc_ids - set(required_passages.keys())
    if missing:
        print(f"  WARNING: {len(missing):,} qrel-referenced passages not found "
              "in corpus stream (may have been removed in a corpus update).")

    # Sample background passages to fill remaining slots
    if max_corpus_size == -1:
        sampled_background = background_pool
    else:
        remaining_slots = max(0, max_corpus_size - len(required_passages))
        random.seed(seed)
        sampled_background = random.sample(
            background_pool,
            min(remaining_slots, len(background_pool))
        )

    corpus_records = list(required_passages.values()) + sampled_background

    print(f"\n  Final corpus size    : {len(corpus_records):,} passages")
    print(f"    qrel-referenced    : {len(required_passages):,}")
    print(f"    background sample  : {len(sampled_background):,}")

    save_jsonl(corpus_records, output_dir / "corpus.jsonl")
    return corpus_records


def save_queries_and_qrels(
    output_dir: Path,
    all_queries: dict,
    qrel_rows: dict,
    corpus_records: list,
):
    """
    Third pass: filter qrels to only reference passages that made it into
    the capped corpus, then save queries and qrels.

    Queries with no surviving qrels are also dropped — they cannot be
    evaluated if their relevant passage was not sampled into the corpus.
    """
    print("\n[3/3] Saving queries and qrels...")

    corpus_ids_in_index = {rec["_id"] for rec in corpus_records}

    surviving_query_ids = set()
    for split, rows in qrel_rows.items():
        filtered = [r for r in rows if r["corpus_id"] in corpus_ids_in_index]
        dropped  = len(rows) - len(filtered)
        if dropped:
            print(f"  WARNING: {dropped} qrel pairs in '{split}' reference "
                  "passages not in the capped corpus and were dropped.")
        save_tsv(
            filtered,
            output_dir / "qrels" / f"{split}.tsv",
            fieldnames=["query_id", "corpus_id", "score"],
        )
        surviving_query_ids.update(r["query_id"] for r in filtered)

    # Only save queries that have at least one surviving qrel
    query_records = [
        {"_id": qid, "text": text}
        for qid, text in all_queries.items()
        if qid in surviving_query_ids
    ]
    save_jsonl(query_records, output_dir / "queries.jsonl")


def verify_output(output_dir: Path, splits: list):
    print("\n" + "=" * 55)
    print("  Download verification")
    print("=" * 55)

    for filename in ["corpus.jsonl", "queries.jsonl"]:
        path = output_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                n = sum(1 for _ in f)
            print(f"  {filename:<22}: {n:,}")
        else:
            print(f"  {filename:<22}: MISSING")

    for split in splits:
        path = output_dir / "qrels" / f"{split}.tsv"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                n = sum(1 for _ in f)
            print(f"  qrels/{split}.tsv{'':>10}: {n:,} relevance pairs")
        else:
            print(f"  qrels/{split}.tsv{'':>10}: MISSING")

    print("=" * 55)
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 55)


def print_pipeline_usage(output_dir: Path):
    corpus  = output_dir.resolve() / "corpus.jsonl"
    queries = output_dir.resolve() / "queries.jsonl"
    qrels   = output_dir.resolve() / "qrels" / "dev.tsv"

    print("\n  Example commands (earlier pipeline, v1_original/main_pipeline.py):\n")
    print(f'  MIRACL_ZH_CORPUS="{corpus}"')
    print(f'  MIRACL_ZH_QUERIES="{queries}"')
    print(f'  MIRACL_ZH_QRELS="{qrels}"')
    print()
    print('  run_experiment "miracl_zh | LDA baseline | gamma=0.7 tau=1.1" \\')
    print('      "experiment_logs/miracl_zh_gamma_0.7_tau_1.1.log" \\')
    print('      python main_pipeline.py --dataset_name "miracl_zh" \\')
    print('      --filepath "$MIRACL_ZH_CORPUS" \\')
    print('      --queries_path "$MIRACL_ZH_QUERIES" \\')
    print('      --qrels_path "$MIRACL_ZH_QRELS" \\')
    print('      --gamma 0.7 --tau 1.1 --num_topics 50')
    print()
    print('  run_experiment "miracl_zh | Lex-TM | gamma=0.7 tau=0.5" \\')
    print('      "experiment_logs/miracl_zh_gamma_0.7_tau_0.5.log" \\')
    print('      python main_pipeline.py --dataset_name "miracl_zh" \\')
    print('      --filepath "$MIRACL_ZH_CORPUS" \\')
    print('      --queries_path "$MIRACL_ZH_QUERIES" \\')
    print('      --qrels_path "$MIRACL_ZH_QRELS" \\')
    print('      --gamma 0.7 --tau 0.5 --num_topics 50')


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit(
            "\nERROR: 'datasets' package not found.\n"
            "Install with:  pip install datasets huggingface_hub\n"
        )

    print("=" * 55)
    print("  MIRACL Chinese — Dataset Download")
    print("=" * 55)
    print(f"  Output dir      : {output_dir.resolve()}")
    print(f"  Splits          : {args.splits}")
    print(f"  Max corpus size : "
          f"{'unlimited' if args.max_corpus_size == -1 else f'{args.max_corpus_size:,}'}")
    print(f"  Random seed     : {args.seed}")

    all_queries, qrel_rows, required_doc_ids = collect_qrel_doc_ids(
        args.splits, args.cache_dir
    )
    corpus_records = download_and_cap_corpus(
        output_dir, required_doc_ids,
        args.max_corpus_size, args.seed, args.cache_dir
    )
    save_queries_and_qrels(
        output_dir, all_queries, qrel_rows, corpus_records
    )
    verify_output(output_dir, args.splits)
    print_pipeline_usage(output_dir)
    print("\nDownload complete.\n")
