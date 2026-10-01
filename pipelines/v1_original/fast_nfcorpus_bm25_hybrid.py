import os
import json
import csv
from beir.retrieval.evaluation import EvaluateRetrieval

# 1. Directly read your local test.tsv
qrels = {}
with open("../data/SOTA/beir/nfcorpus/qrels/test.tsv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader) # Skip the header
    for row in reader:
        if len(row) >= 3:
            q_id, d_id, score = row[0], row[1], int(row[2])
            if q_id not in qrels:
                qrels[q_id] = {}
            qrels[q_id][d_id] = score

# 2. Load the run files saved by run_trec_covid.py / run_nfcorpus.py
print("Loading cached results...")
with open("result/nfcorpus_raw_results_gamma_1.0.json", "r", encoding="utf-8") as f:
    dense_results = json.load(f)

with open("result/nfcorpus_raw_results_bm25.json", "r", encoding="utf-8") as f:
    bm25_results = json.load(f)

# 3. Calculate Scores (Min-Max Interpolation AND RRF)
print("Fusing Scores...")
hybrid_results = {}
rrf_results = {}
k_constant = 60

for qid in qrels.keys():
    hybrid_results[qid] = {}
    rrf_results[qid] = {}
    
    dense_scores = dense_results.get(qid, {})
    bm25_scores = bm25_results.get(qid, {})
    
    # --- Setup for Min-Max ---
    d_min = min(dense_scores.values()) if dense_scores else 0
    d_max = max(dense_scores.values()) if dense_scores else 1
    b_min = min(bm25_scores.values()) if bm25_scores else 0
    b_max = max(bm25_scores.values()) if bm25_scores else 1
    
    # --- Setup for RRF (Sorting to get Ranks) ---
    dense_ranked = sorted(dense_scores.keys(), key=lambda x: dense_scores[x], reverse=True)
    bm25_ranked = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True)
    
    dense_rank_dict = {doc_id: rank for rank, doc_id in enumerate(dense_ranked, 1)}
    bm25_rank_dict = {doc_id: rank for rank, doc_id in enumerate(bm25_ranked, 1)}
    
    all_docs = set(dense_scores.keys()).union(set(bm25_scores.keys()))
    
    for doc_id in all_docs:
        # Min-Max Math
        raw_d = dense_scores.get(doc_id, d_min)
        norm_d = (raw_d - d_min) / (d_max - d_min) if d_max > d_min else 0.0
        
        raw_b = bm25_scores.get(doc_id, b_min)
        norm_b = (raw_b - b_min) / (b_max - b_min) if b_max > b_min else 0.0
        
        hybrid_results[qid][doc_id] = (0.5 * norm_d) + (0.5 * norm_b)
        
        # RRF Math
        d_rank = dense_rank_dict.get(doc_id, float('inf'))
        b_rank = bm25_rank_dict.get(doc_id, float('inf'))
        
        rrf_d = 1.0 / (k_constant + d_rank) if d_rank != float('inf') else 0.0
        rrf_b = 1.0 / (k_constant + b_rank) if b_rank != float('inf') else 0.0
        
        rrf_results[qid][doc_id] = rrf_d + rrf_b

# 4. Evaluate using official BEIR tool
print("\n=== NFCORPUS FINAL METRICS ===")
ndcg_bm25, _, _, _ = EvaluateRetrieval.evaluate(qrels, bm25_results, [10])
print(f"  Pure BM25 NDCG@10:       {ndcg_bm25['NDCG@10']:.4f}")

ndcg_hybrid, _, _, _ = EvaluateRetrieval.evaluate(qrels, hybrid_results, [10])
print(f"  Min-Max Hybrid NDCG@10:  {ndcg_hybrid['NDCG@10']:.4f}")

ndcg_rrf, _, _, _ = EvaluateRetrieval.evaluate(qrels, rrf_results, [10])
print(f"  RRF NDCG@10:             {ndcg_rrf['NDCG@10']:.4f}")
