"""MedWeb statistics in Sections 6.2-6.4 of the paper, computed with the
pipeline's own loader and tokenizer: Spearman correlation of HHI with document
frequency; share of term-document pairs with count one; number of terms
amplified at tau = 0.5 / 0.8 / 1.1 and their document frequencies; raw tweet
lengths. It also recomputes Pure BM25 with rank_bm25 (as run, best/worst tie
resolution, label-free index) as a cross-check; the paper's BM25 numbers come
from results/camera_ready/medweb_results.json.

usage: python scripts/analysis/medweb_prior_stats.py --medweb_dir <dir> [--out medweb_prior_stats.json]
"""
import argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipelines", "v3_patched"))
import json, numpy as np, pandas as pd, jieba
from rank_bm25 import BM25Okapi
from collections import Counter
from rag_dataload import UniversalRAGLoader
jieba.setLogLevel(60)
ap = argparse.ArgumentParser(description="MedWeb statistics of Sections 6.2-6.4")
ap.add_argument("--medweb_dir", required=True)
ap.add_argument("--out", default="medweb_prior_stats.json")
args = ap.parse_args()
D = args.medweb_dir

def build_eval(df, chunks):
    lookup = {}
    for c in chunks: lookup.setdefault(c['doc_id'], set()).add(c['chunk_id'])
    groups = {}
    for _, r in df.iterrows():
        q = str(r['ReviewTitle']).strip()
        if q and r['document_id'] in lookup:
            groups.setdefault(q, set()).update(lookup[r['document_id']])
    return [{'query': q, 'gt': g} for q, g in groups.items()]

def rr_from_scores(s, ids, gt, mode, top_k=20):
    if mode == 'argsort':            # exactly what the pipeline does
        order = np.argsort(s)[::-1][:top_k]
    else:                            # optimistic / pessimistic tie resolution
        rel = np.array([i in gt for i in ids])
        key = rel.astype(float) if mode == 'best' else -rel.astype(float)
        order = np.lexsort((-key, -s))[:top_k]
    rank = next((i+1 for i, j in enumerate(order) if ids[j] in gt), 0)
    return 1.0/rank if rank else 0.0

res = {}
for lang in ['en', 'ja', 'zh']:
    L = UniversalRAGLoader(chunk_size=512, chunk_size_zh=512, chunk_size_ja=512, overlap=73, min_df=2)
    df = L.load_medweb(f'{D}/medweb_rag_{lang}_fixed.csv', lang)
    raw = pd.read_csv(f'{D}/medweb_rag_{lang}_fixed.csv', encoding='utf-8-sig')
    raw['document_id'] = 'medweb_' + raw['document_id'].astype(str)
    df = df.merge(raw[['document_id', 'ReviewText']], on='document_id')
    chunks, bow, da = L.process_corpus(df[['document_id','ReviewTitle','full_text']], language=lang)
    ev = build_eval(df, chunks)
    ids = [c['chunk_id'] for c in chunks]
    bm = BM25Okapi([L.clean_and_tokenize(c['text'], lang) for c in chunks])
    out = {m: [] for m in ['argsort', 'best', 'worst']}
    per_q = []
    for e in ev:
        s = bm.get_scores(L.clean_and_tokenize(e['query'], lang))
        for m in out: out[m].append(rr_from_scores(s, ids, e['gt'], m))
        per_q.append((e['query'], len(e['gt']), out['argsort'][-1]))
    # label-free variant
    L2 = UniversalRAGLoader(chunk_size=512, chunk_size_zh=512, chunk_size_ja=512, overlap=73, min_df=2)
    df2 = df.copy(); df2['full_text'] = df2['ReviewText'].astype(str)
    ch2, _, _ = L2.process_corpus(df2[['document_id','ReviewTitle','full_text']], language=lang)
    ev2 = build_eval(df2, ch2); ids2 = [c['chunk_id'] for c in ch2]
    bm2 = BM25Okapi([L2.clean_and_tokenize(c['text'], lang) for c in ch2])
    rr2 = [rr_from_scores(bm2.get_scores(L2.clean_and_tokenize(e['query'], lang)), ids2, e['gt'], 'argsort') for e in ev2]
    # HHI (document scope, as in lex_tm_model) vs df
    V = L.vocab_size
    cnt = np.zeros(V); dfv = np.zeros(V); sq = np.zeros(V)
    for d in bow:
        for w, c in d:
            cnt[w] += c; dfv[w] += 1; sq[w] += c*c
    hhi = sq / cnt**2
    E = (hhi - hhi.min()) / (hhi.max() - hhi.min())
    singleton = sum(1 for d in bow for w, c in d if c == 1) / sum(1 for d in bow for w, c in d)
    from scipy.stats import spearmanr
    amp = {str(t): int((E > t).sum()) for t in (0.5, 0.8, 1.1)}
    amp_df = {str(t): {str(k): int(v) for k, v in Counter(dfv[E > t].astype(int)).most_common(4)} for t in (0.5, 0.8)}
    # raw lengths (before stopword/length filtering)
    if lang == 'en':
        rawlen = [len(str(t).split()) for t in df['ReviewText']]
    elif lang == 'zh':
        rawlen = [len([w for w in jieba.lcut(str(t)) if w.strip()]) for t in df['ReviewText']]
    else:
        rawlen = [len(L._ja_tagger(str(t))) for t in df['ReviewText']]
    chars = [len(str(t)) for t in df['ReviewText']]
    res[lang] = dict(vocab=V, n_q=len(ev), bm25_as_run=np.mean(out['argsort']),
        bm25_best_ties=np.mean(out['best']), bm25_worst_ties=np.mean(out['worst']),
        bm25_label_free=np.mean(rr2),
        spearman_hhi_df=float(spearmanr(hhi, dfv).correlation), singleton_frac=singleton,
        amplified=amp, amplified_df_hist=amp_df,
        tweet_raw_tokens_mean=float(np.mean(rawlen)), tweet_chars_mean=float(np.mean(chars)))
    print(lang, json.dumps(res[lang], default=float), flush=True)
    if lang == 'en':
        for q, n, r in sorted(per_q, key=lambda x: -x[1])[:39]: print(f"   {n:4d}  RR={r:.3f}  {q}")
json.dump(res, open(args.out, 'w'), indent=1, default=float)
print('wrote', args.out)
