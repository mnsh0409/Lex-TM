import os
import string
import pandas as pd
from tqdm import tqdm
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# Earlier-pipeline modules (this directory)
from rag_dataload import UniversalRAGLoader
from lex_tm_model import LexTMLdaModel
from rag_evaluation_pipeline import LexTMRouter

def tokenize_for_bm25(loader, text):
    """English branch of UniversalRAGLoader.tokenize_for_bm25() in the loader
    version used for this run (not in v1_original/rag_dataload.py): lowercase,
    strip ASCII punctuation, split on whitespace, drop NLTK stopwords; numbers
    and short words are kept."""
    if not isinstance(text, str):
        return []
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return [w for w in text.split() if w not in loader.en_stopwords]


def main():
    print("=== Step 1: Downloading TREC-COVID via BEIR ===")
    #dataset = "trec-covid"
    #url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    #out_dir = os.path.join(os.getcwd(), "datasets")
    #data_path = util.download_and_unzip(url, out_dir)
    data_path = "../data/SOTA/beir/trec-covid"

    print("=== Step 2: Loading Data into Pandas for Lex-TM Pipeline ===")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    # Convert the BEIR corpus to the format UniversalRAGLoader expects
    records = []
    for doc_id, doc_data in corpus.items():
        title = doc_data.get("title", "")
        text = doc_data.get("text", "")
        records.append({
            "document_id": str(doc_id),
            "full_text": f"{title}. {text}".strip()
        })
    
    df = pd.DataFrame(records)
    
    
    print(f"Loaded {len(df)} documents for processing.")

    print("\n=== Step 3: Lex-TM Preprocessing ===")
    loader = UniversalRAGLoader(
        chunk_size=150, 
        overlap=30, 
        min_df=2
    )
    all_chunks_dicts, corpus_bow, doc_assignments = loader.process_corpus(df, language='en')

    print("\n=== Step 4: Lex-TM Gibbs Training ===")
    lex_tm = LexTMLdaModel(
        num_topics=50, 
        alpha=0.1, 
        tau=0.5, 
        lambda_amp=2.0, 
        random_state=42
    )
    lex_tm.fit(
        corpus=corpus_bow,
        dictionary=loader.id2token,
        max_iter=300, # 300 sweeps in these runs (500 elsewhere)
        convergence_threshold=1e-4,
        min_iter=50,
        doc_assignments=doc_assignments,
        hhi_scope='document'
    )

    print("\n=== Step 5: BGE-m3 Dense Indexing & Routing ===")
    router = LexTMRouter(
        dense_model_name='BAAI/bge-m3',
        lex_tm_model=lex_tm,
        dictionary=loader,
        language='en',
        min_query_tokens=3 
    )
    router.index_corpus(all_chunks_dicts, cache_path="cache_trec_covid")

    print("\n=== Step 6: Retrieving and Formatting for BEIR ===")
    # Two setups: Pure Dense (baseline) and Lex-TM
    gammas_to_test = [1.0, 0.7] # 1.0 = Pure Dense baseline, 0.7 = Lex-TM

    for gamma in gammas_to_test:
        print(f"\n--- Evaluating Gamma = {gamma} ---")
        lex_tm_results = {}
        
        for qid, query_text in tqdm(queries.items(), desc=f"Routing queries (g={gamma})"):
            retrieved_chunks = router.retrieve(query_text, top_k=100, gamma=gamma)
            
            # BEIR evaluates DOCUMENTS, not chunks. 
            # We take the max chunk score for each document (standard RAG practice).
            doc_scores = {}
            for chunk in retrieved_chunks:
                did = chunk['doc_id']
                score = chunk['final_score']
                if did not in doc_scores or score > doc_scores[did]:
                    doc_scores[did] = score
            
            lex_tm_results[qid] = doc_scores

        print(f"\n=== Step 7: Calculating Official Metrics for Gamma {gamma} ===")
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, lex_tm_results, [10, 100])
        
        # 1. Print and save the metrics
        result_text = f"RESULTS FOR GAMMA {gamma}:\n  NDCG@10: {ndcg['NDCG@10']:.4f}\n  Recall@100: {recall['Recall@100']:.4f}\n"
        print(result_text)
        
        os.makedirs("result", exist_ok=True)
        with open("result/trec_covid_final_metrics.txt", "a") as f:
            f.write(result_text + "\n")
            
        # 2. Save the raw run file (used by fast_*_bm25_hybrid.py)
        import json
        with open(f"result/trec_covid_raw_results_gamma_{gamma}.json", "w") as f:
            json.dump(lex_tm_results, f)

    # ==========================================
    # BM25 BLOCK (With Saving Added)
    # ==========================================
    print("\n=== EXTRA STEP: Calculating Pure BM25 Baseline ===")
    from rank_bm25 import BM25Okapi
    import numpy as np
    import json

    print("Tokenizing corpus for BM25...")
    corpus_texts = df['full_text'].tolist()
    doc_ids = df['document_id'].tolist()
    tokenized_corpus = [
        tokenize_for_bm25(loader, str(text))
        for text in corpus_texts
    ] #[str(text).lower().split() for text in corpus_texts]
    
    print("Building BM25 Index...")
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_results = {}
    
    for qid, query_text in tqdm(queries.items(), desc="Routing queries (BM25)"):
        tokenized_query = str(query_text).lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:100]
        bm25_results[qid] = {str(doc_ids[idx]): float(scores[idx]) for idx in top_k_indices}

    print("\n=== BM25 Official Metrics ===")
    ndcg_bm25, _, recall_bm25, _ = EvaluateRetrieval.evaluate(qrels, bm25_results, [10, 100])
    
    bm25_text = f"RESULTS FOR BM25:\n  NDCG@10: {ndcg_bm25['NDCG@10']:.4f}\n  Recall@100: {recall_bm25['Recall@100']:.4f}\n"
    print(bm25_text)
    
    with open("result/trec_covid_final_metrics.txt", "a") as f:
        f.write(bm25_text + "\n")
        
    with open("result/trec_covid_raw_results_bm25.json", "w") as f:
        json.dump(bm25_results, f)

if __name__ == "__main__":
    main()
