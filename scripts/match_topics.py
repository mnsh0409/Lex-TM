#!/usr/bin/env python3
"""Pair the topics of two topic models by shared top words (paper Table 4).

usage: python scripts/match_topics.py <lda_routing_topics.json> <lextm_topics.json> [--top 10]

Each file is the list of per-topic top-word lists that main_pipeline1_patched.py
writes to experiment_logs1/<dataset>_gamma_<g>_tau_<t>_topics.json. For every
topic of the first model (LDA routing, tau = 1.1) the script prints the topic of
the second model (Lex-TM, tau = 0.8) that shares the most of its top words.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--top", type=int, default=10, help="top words compared per topic")
    args = ap.parse_args()
    with open(args.first, encoding="utf-8") as f:
        a = [t[: args.top] for t in json.load(f)]
    with open(args.second, encoding="utf-8") as f:
        b = [t[: args.top] for t in json.load(f)]
    best = []
    for i, ta in enumerate(a):
        overlaps = [len(set(ta) & set(tb)) for tb in b]
        j = max(range(len(b)), key=lambda k: overlaps[k])
        best.append(overlaps[j])
        print(f"topic {i:>2} <-> {j:>2}  shared {overlaps[j]}/{args.top}")
        print(f"   first : {', '.join(ta)}")
        print(f"   second: {', '.join(b[j])}")
    print(f"mean shared top words over best-matched pairs: {sum(best) / len(best):.1f} of {args.top}")


if __name__ == "__main__":
    main()
