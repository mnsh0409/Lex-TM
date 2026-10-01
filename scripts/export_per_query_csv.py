#!/usr/bin/env python3
"""Write per_query/medweb_<lang>.csv from medweb_results.json.

usage: python scripts/export_per_query_csv.py <results dir>   (e.g. results/camera_ready)

One row per label set: query, n_relevant (relevant tweets), then the reciprocal
rank (MRR@20 per query) of every system, in the order of langs.<lang>.queries.
"""
import csv
import json
import os
import sys


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/camera_ready"
    with open(os.path.join(out, "medweb_results.json"), encoding="utf-8") as f:
        d = json.load(f)
    os.makedirs(os.path.join(out, "per_query"), exist_ok=True)
    for lang, r in d["langs"].items():
        systems = list(r["per_query"])
        path = os.path.join(out, "per_query", f"medweb_{lang}.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["query", "n_relevant"] + systems)
            for i, q in enumerate(r["queries"]):
                w.writerow([q, r["group_sizes"][i]] + [f"{r['per_query'][s][i]:.6f}" for s in systems])
        print("wrote", path)


if __name__ == "__main__":
    main()
