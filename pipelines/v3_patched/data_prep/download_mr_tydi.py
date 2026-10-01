"""
download_mr_tydi.py
===================
Downloads a Mr. TyDi language subset from HuggingFace and saves it in a
format directly compatible with the Lex-TM pipeline loader.

Mr. TyDi (Multilingual Tydi Question Answering) is a multilingual IR
benchmark based on TyDi QA, with Wikipedia passage corpora and human-
annotated relevance judgements.

Available languages
-------------------
arabic, bengali, english, indonesian, finnish, korean, russian, swahili,
telugu, thai, japanese

This script is used for Japanese ('ja') and Thai ('th') in the Lex-TM
evaluation suite. Both are non-alphabetic (or space-free) scripts where
the HHI exclusivity heuristic's token-agnostic claim is most meaningful.

HuggingFace sources
-------------------
Queries + qrels : castorini/mr-tydi        (language=<lang>)
Corpus          : castorini/mr-tydi-corpus (language=<lang>)

License: Apache 2.0

Corpus size note
----------------
Japanese corpus: ~7M Wikipedia passages — must be capped.
Thai corpus    : ~568K passages          — must be capped.
Default cap: 100K passages, all qrel-referenced passages retained (seed=42).

Output structure
----------------
data/mr_tydi_<lang>/
    corpus.jsonl          {_id, title, text}
    queries.jsonl         {_id, text}
    qrels/test.tsv        query_id, corpus_id, score (tab-separated)
    qrels/dev.tsv         same (if --splits includes dev)

This mirrors the MIRACL format so load_mr_tydi() and load_miracl_zh()
share the same JSONL reading code.

Dependencies
------------
    pip install datasets huggingface_hub
    pip install fugashi ipadic    # Japanese tokenization (for pipeline)
    pip install pythainlp         # Thai tokenization (for pipeline)

Usage
-----
    python download_mr_tydi.py --language japanese --output_dir ../data/mr_tydi_ja
    python download_mr_tydi.py --language thai     --output_dir ../data/mr_tydi_th
    python download_mr_tydi.py --language japanese --max_corpus_size 50000
"""

import argparse
import json
import csv
import random
from pathlib import Path

# Maps our shorthand language codes to Mr. TyDi HuggingFace config names
_LANG_CODE_TO_TYDI = {
    'ja': 'japanese',
    'th': 'thai',
    'ar': 'arabic',
    'bn': 'bengali',
    'en': 'english',
    'id': 'indonesian',
    'fi': 'finnish',
    'ko': 'korean',
    'ru': 'russian',
    'sw': 'swahili',
    'te': 'telugu',
}

# Inverse map: tydi name → short code (used to derive id_prefix)
_TYDI_TO_LANG_CODE = {v: k for k, v in _LANG_CODE_TO_TYDI.items()}

_VALID_LANGUAGES = list(_LANG_CODE_TO_TYDI.keys()) + list(_LANG_CODE_TO_TYDI.values())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Mr. TyDi language subset from HuggingFace."
    )
    parser.add_argument(
        "--language", type=str, required=True,
        help=(
            "Language to download. Accepts short codes ('ja', 'th') or "
            "full names ('japanese', 'thai'). "
            f"Available: {', '.join(_LANG_CODE_TO_TYDI.keys())} "
            f"(or full names: {', '.join(_LANG_CODE_TO_TYDI.values())})"
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Directory to save the dataset. "
            "Defaults to ../data/mr_tydi_<lang_code> "
            "(e.g. ../data/mr_tydi_ja for Japanese)."
        ),
    )
    parser.add_argument(
        "--splits", nargs="+", default=["test"],
        choices=["train", "dev", "test"],
        help="Query/qrel splits to download. Default: test.",
    )
    parser.add_argument(
        "--max_corpus_size", type=int, default=100_000,
        help=(
            "Maximum number of corpus passages to retain. "
            "All qrel-referenced passages are always kept. "
            "Default: 100000. Set to -1 for full corpus (very slow for Japanese)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for background passage sampling. Default: 42.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HuggingFace cache directory. Defaults to ~/.cache/huggingface.",
    )
    return parser.parse_args()


def resolve_language(language_arg: str):
    """
    Accepts both short codes ('ja') and full names ('japanese').
    Returns (short_code, tydi_name, display_name).
    """
    lang = language_arg.lower().strip()

    if lang in _LANG_CODE_TO_TYDI:
        short_code = lang
        tydi_name  = _LANG_CODE_TO_TYDI[lang]
    elif lang in _TYDI_TO_LANG_CODE:
        tydi_name  = lang
        short_code = _TYDI_TO_LANG_CODE[lang]
    else:
        raise ValueError(
            f"Unknown language '{language_arg}'. "
            f"Valid options: {', '.join(_VALID_LANGUAGES)}"
        )

    display_name = tydi_name.capitalize()
    return short_code, tydi_name, display_name


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


def collect_qrel_doc_ids(tydi_name: str, splits: list, cache_dir: str) -> tuple:
    """
    First pass: load queries and qrels, collect corpus doc IDs referenced
    in relevance judgements. These are always included in the corpus.

    Returns
    -------
    all_queries     : dict {query_id: query_text}
    qrel_rows       : dict {split: list of {query_id, corpus_id, score}}
    required_doc_ids: set of doc IDs referenced in any qrel
    """
    from datasets import load_dataset

    print(f"\n[1/3] Loading queries and qrels for splits: {splits}")

    all_queries = {}
    qrel_rows = {split: [] for split in splits}
    required_doc_ids = set()

    for split in splits:
        print(f"  Fetching split: {split} ...")
        dataset = load_dataset(
            "castorini/mr-tydi",
            tydi_name,
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

            # Include hard negatives in corpus so retrieval is non-trivial
            for passage in row.get("negative_passages", []):
                required_doc_ids.add(str(passage["docid"]))

        if skipped:
            print(f"  WARNING: {skipped} queries had no positive passages.")

    print(f"\n  Total unique queries  : {len(all_queries):,}")
    print(f"  Total qrel pairs      : "
          f"{sum(len(v) for v in qrel_rows.values()):,}")
    print(f"  Required corpus IDs   : {len(required_doc_ids):,}")

    return all_queries, qrel_rows, required_doc_ids


def download_and_cap_corpus(
    output_dir: Path,
    tydi_name: str,
    required_doc_ids: set,
    max_corpus_size: int,
    seed: int,
    cache_dir: str,
) -> list:
    """
    Second pass: stream the Mr. TyDi corpus, always keep qrel-referenced
    passages, fill remaining slots with a random sample.

    Mr. TyDi corpus fields: docid, title, text
    Normalised to:          _id,   title, text  (MIRACL-compatible format)
    """
    from datasets import load_dataset

    print(f"\n[2/3] Downloading corpus (castorini/mr-tydi-corpus, {tydi_name})...")

    if max_corpus_size == -1:
        print("  max_corpus_size=-1: downloading full corpus. May be very slow.")
    else:
        print(f"  Corpus cap          : {max_corpus_size:,} passages")
        print(f"  Guaranteed          : {len(required_doc_ids):,} qrel-referenced")

    corpus_stream = load_dataset(
        "castorini/mr-tydi-corpus",
        tydi_name,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    required_passages = {}
    background_pool   = []

    print("  Streaming corpus passages...")
    processed = 0

    for row in corpus_stream:
        doc_id = str(row["docid"])   # Mr. TyDi uses "docid", not "_id"
        record = {
            "_id":   doc_id,         # normalise to MIRACL-compatible key
            "title": row.get("title", ""),
            "text":  row["text"],
        }

        if doc_id in required_doc_ids:
            required_passages[doc_id] = record
        elif max_corpus_size == -1:
            background_pool.append(record)
        else:
            remaining_slots = max_corpus_size - len(required_doc_ids)
            if len(background_pool) < remaining_slots * 5:
                background_pool.append(record)

        processed += 1
        if processed % 500_000 == 0:
            print(f"  Processed {processed:,} passages "
                  f"(required found: {len(required_passages):,})...")

        # Early exit once sufficient background passages collected
        if (max_corpus_size != -1
                and len(required_passages) == len(required_doc_ids)
                and len(background_pool) >= (
                    max_corpus_size - len(required_doc_ids)) * 3):
            print(f"  Early exit at {processed:,} — sufficient pool collected.")
            break

    print(f"\n  Required passages found : {len(required_passages):,} "
          f"/ {len(required_doc_ids):,}")

    missing = required_doc_ids - set(required_passages.keys())
    if missing:
        print(f"  WARNING: {len(missing):,} qrel-referenced passages not found "
              "in corpus stream.")

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

    print(f"\n  Final corpus size     : {len(corpus_records):,} passages")
    print(f"    qrel-referenced     : {len(required_passages):,}")
    print(f"    background sample   : {len(sampled_background):,}")

    save_jsonl(corpus_records, output_dir / "corpus.jsonl")
    return corpus_records


def save_queries_and_qrels(
    output_dir: Path,
    all_queries: dict,
    qrel_rows: dict,
    corpus_records: list,
):
    """
    Third pass: filter qrels to passages that made it into the capped corpus,
    drop queries with no surviving qrels, then save everything.
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
            print(f"  qrels/{split}.tsv{'':>12}: {n:,} relevance pairs")
        else:
            print(f"  qrels/{split}.tsv{'':>12}: MISSING")

    print("=" * 55)
    print(f"  Output: {output_dir.resolve()}")
    print("=" * 55)


def print_pipeline_usage(output_dir: Path, short_code: str, splits: list):
    qrels_split = "test" if "test" in splits else splits[0]
    corpus  = output_dir.resolve() / "corpus.jsonl"
    queries = output_dir.resolve() / "queries.jsonl"
    qrels   = output_dir.resolve() / "qrels" / f"{qrels_split}.tsv"
    dataset = f"mr_tydi_{short_code}"

    print(f"\n  Example commands (earlier pipeline, v1_original/main_pipeline.py):\n")
    print(f'  MR_TYDI_{short_code.upper()}_CORPUS="{corpus}"')
    print(f'  MR_TYDI_{short_code.upper()}_QUERIES="{queries}"')
    print(f'  MR_TYDI_{short_code.upper()}_QRELS="{qrels}"')
    print()
    print(f'  run_experiment "{dataset} | LDA baseline | gamma=0.7 tau=1.1" \\')
    print(f'      "experiment_logs/{dataset}_gamma_0.7_tau_1.1.log" \\')
    print(f'      python main_pipeline.py --dataset_name "{dataset}" \\')
    print(f'      --filepath "$MR_TYDI_{short_code.upper()}_CORPUS" \\')
    print(f'      --queries_path "$MR_TYDI_{short_code.upper()}_QUERIES" \\')
    print(f'      --qrels_path "$MR_TYDI_{short_code.upper()}_QRELS" \\')
    print(f'      --gamma 0.7 --tau 1.1 --num_topics 50')
    print()
    print(f'  run_experiment "{dataset} | Lex-TM | gamma=0.7 tau=0.5" \\')
    print(f'      "experiment_logs/{dataset}_gamma_0.7_tau_0.5.log" \\')
    print(f'      python main_pipeline.py --dataset_name "{dataset}" \\')
    print(f'      --filepath "$MR_TYDI_{short_code.upper()}_CORPUS" \\')
    print(f'      --queries_path "$MR_TYDI_{short_code.upper()}_QUERIES" \\')
    print(f'      --qrels_path "$MR_TYDI_{short_code.upper()}_QRELS" \\')
    print(f'      --gamma 0.7 --tau 0.5 --num_topics 50')


if __name__ == "__main__":
    args = parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit(
            "\nERROR: 'datasets' package not found.\n"
            "Install with:  pip install datasets huggingface_hub\n"
        )

    short_code, tydi_name, display_name = resolve_language(args.language)

    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else f"../data/mr_tydi_{short_code}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"  Mr. TyDi {display_name} — Dataset Download")
    print("=" * 55)
    print(f"  Language        : {display_name} ({short_code})")
    print(f"  Output dir      : {output_dir.resolve()}")
    print(f"  Splits          : {args.splits}")
    print(f"  Max corpus size : "
          f"{'unlimited' if args.max_corpus_size == -1 else f'{args.max_corpus_size:,}'}")
    print(f"  Random seed     : {args.seed}")

    all_queries, qrel_rows, required_doc_ids = collect_qrel_doc_ids(
        tydi_name, args.splits, args.cache_dir
    )
    corpus_records = download_and_cap_corpus(
        output_dir, tydi_name, required_doc_ids,
        args.max_corpus_size, args.seed, args.cache_dir
    )
    save_queries_and_qrels(
        output_dir, all_queries, qrel_rows, corpus_records
    )
    verify_output(output_dir, args.splits)
    print_pipeline_usage(output_dir, short_code, args.splits)
    print("\nDownload complete.\n")
