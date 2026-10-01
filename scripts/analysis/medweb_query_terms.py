"""Query-term statistics of Section 6.2 and label-set sizes of Sections 4.1 and
6.3: distinct query terms per language, the smallest document frequency of a
query term, the largest exclusivity score E_v of a query term, query terms
amplified at tau = 0.8 / 0.5, and the distribution of label-set sizes and
label counts.

usage: python scripts/analysis/medweb_query_terms.py --medweb_dir <dir>
"""
import argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipelines", "v3_patched"))
import numpy as np, jieba, json
jieba.setLogLevel(60)
import camera_ready_extras as cre
ap = argparse.ArgumentParser(description="MedWeb query-term statistics")
ap.add_argument("--medweb_dir", required=True)
D = ap.parse_args().medweb_dir
out={}
for lang in ['en','ja','zh']:
    L=cre.make_loader(); df=L.load_medweb(f'{D}/medweb_rag_{lang}_fixed.csv',lang)
    ch,bow,da=L.process_corpus(df,language=lang); ev=cre.title_groups(df,ch)
    V=L.vocab_size; cnt=np.zeros(V); dfv=np.zeros(V); sq=np.zeros(V)
    for d in bow:
        for w,c in d: cnt[w]+=c; dfv[w]+=1; sq[w]+=c*c
    hhi=sq/cnt**2; E=(hhi-hhi.min())/(hhi.max()-hhi.min())
    qtok=set(t for e in ev for t in L.clean_and_tokenize(e['query'],lang))
    qids=[L.token2id[t] for t in qtok if t in L.token2id]
    sizes=sorted([len(e['gt']) for e in ev])
    nl=[cre.n_labels(e['query']) for e in ev]
    out[lang]=dict(query_vocab=len(qtok), query_terms_in_vocab=len(qids),
        max_E_query=float(E[qids].max()), min_df_query=int(dfv[qids].min()),
        amplified_query_terms_tau08=int((E[qids]>0.8).sum()), amplified_query_terms_tau05=int((E[qids]>0.5).sum()),
        groups=len(ev), size_min=sizes[0], size_median=float(np.median(sizes)), size_max=sizes[-1],
        groups_size1=sum(1 for s in sizes if s==1), groups_lt10=sum(1 for s in sizes if s<10), groups_ge100=sum(1 for s in sizes if s>=100),
        n_labels_hist={str(k):nl.count(k) for k in sorted(set(nl))},
        docs_in_groups_ge100=sum(s for s in sizes if s>=100))
    print(lang, json.dumps(out[lang], ensure_ascii=False), flush=True)
