"""Document and query lengths of NFCorpus, TREC-COVID (20,000-document sample)
and ChatDoctor quoted in Section 6.3 (content tokens after the pipeline's
English tokenizer, and raw words).

usage: python scripts/analysis/beir_length_stats.py --beir_dir <dir with nfcorpus/ and trec-covid/> --chatdoctor_dir <dir>
"""
import argparse
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pipelines", "v3_patched"))
import json, numpy as np, random
from rag_dataload import UniversalRAGLoader
L=UniversalRAGLoader(chunk_size=512, overlap=73, min_df=2)
def stats(corpus_path, queries_path, qrels_path, sample=None):
    docs=[json.loads(l) for l in open(corpus_path,encoding='utf-8') if l.strip()]
    if sample and len(docs)>sample:
        random.seed(0); docs=random.sample(docs,sample)
    dl=[len(L.clean_and_tokenize((d.get('title','')+' '+d['text']).strip(),'en')) for d in docs]
    rawl=[len((d.get('title','')+' '+d['text']).split()) for d in docs]
    qids=set(l.split('\t')[0] for l in open(qrels_path,encoding='utf-8') if l.strip() and not l.startswith('query'))
    qs=[json.loads(l) for l in open(queries_path,encoding='utf-8') if l.strip()]
    ql=[len(L.clean_and_tokenize(q['text'],'en')) for q in qs if str(q['_id']) in qids]
    return dict(n_docs_scored=len(docs), doc_content_tokens=round(float(np.mean(dl)),1), doc_words=round(float(np.mean(rawl)),1),
                n_queries=len(ql), query_content_tokens=round(float(np.mean(ql)),1), queries_lt5=round(float(np.mean([x<5 for x in ql])),3))
ap = argparse.ArgumentParser(description="BEIR length statistics")
ap.add_argument("--beir_dir", required=True)
ap.add_argument("--chatdoctor_dir", required=True)
args = ap.parse_args()
B = args.beir_dir
print('nfcorpus', stats(f'{B}/nfcorpus/corpus.jsonl', f'{B}/nfcorpus/queries.jsonl', f'{B}/nfcorpus/qrels/test.tsv'))
print('trec-covid', stats(f'{B}/trec-covid/corpus.jsonl', f'{B}/trec-covid/queries.jsonl', f'{B}/trec-covid/qrels/test.tsv', sample=20000))
C = args.chatdoctor_dir
print('chatdoctor', stats(f'{C}/corpus.jsonl', f'{C}/queries.jsonl', f'{C}/qrels/test.tsv'))
