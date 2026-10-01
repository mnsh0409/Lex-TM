import os
import argparse
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Import your existing pipeline components
from rag_dataload import UniversalRAGLoader
from main_pipeline1 import (
    _DATASET_LANG, 
    _QRELS_DATASETS, 
    QUERY_COLUMN,
    load_queries_and_qrels, 
    build_evaluation_dataset, 
    build_evaluation_dataset_qrels
)
from rag_evaluation_pipeline import evaluate_retrieval

class BM25HybridRouter:
    """
    Drop-in replacement for LexTMRouter that evaluates BM25, 
    Pure Dense (BGE-m3), and a normalized Hybrid combination.
    """
    def __init__(self, dense_model_name: str, dictionary: UniversalRAGLoader, language: str):
        print(f"Initializing BM25HybridRouter (Dense: {dense_model_name}, Lang: {language})")
        self.dense_model = SentenceTransformer(dense_model_name)
        self.dictionary = dictionary
        self.language = language
        self.bm25 = None
        self.corpus_chunks = []
        self.dense_embeddings = None

    def index_corpus(self, chunks: list, cache_path: str = None):
        self.corpus_chunks = chunks
        
        # 1. Build BM25 Index using your exact CJK/English tokenizers
        print("Tokenizing corpus for BM25...")
        tokenized_corpus = []
        for c in tqdm(chunks, desc="Tokenizing"):
            tokens = self.dictionary.clean_and_tokenize(c['text'], self.language)
            tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Load or Compute Dense Embeddings (Reuses your Lex-TM cache!)
        emb_cache_path = f"{cache_path}_embeddings.npy" if cache_path else None
        
        if emb_cache_path and os.path.exists(emb_cache_path):
            print(f"Loading cached dense embeddings from {emb_cache_path}")
            self.dense_embeddings = torch.tensor(np.load(emb_cache_path), dtype=torch.float32)
        else:
            print("Computing BGE-m3 dense embeddings...")
            texts = [c['text'] for c in chunks]
            self.dense_embeddings = self.dense_model.encode(
                texts, batch_size=256, convert_to_tensor=True, normalize_embeddings=True
            )
            if emb_cache_path:
                np.save(emb_cache_path, self.dense_embeddings.cpu().numpy())
                
    def retrieve(self, query: str, top_k: int = 20, gamma: float = 0.5) -> list:
        """
        gamma acts as the dense weight (alpha):
        gamma = 1.0 -> Pure Dense BGE-m3
        gamma = 0.0 -> Pure BM25
        gamma = 0.5 -> 50/50 Hybrid
        """
        # 1. BM25 Scores
        tokenized_query = self.dictionary.clean_and_tokenize(query, self.language)
        bm25_raw = self.bm25.get_scores(tokenized_query)
        
        # 2. Dense Scores (Cosine Similarity)
        query_emb = self.dense_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        dense_raw = torch.matmul(self.dense_embeddings, query_emb).cpu().numpy()
        dense_raw = (dense_raw + 1.0) / 2.0  # Scale [-1, 1] to [0, 1]
        
        # 3. Min-Max Normalize BM25 per query for fair interpolation
        bm25_min, bm25_max = np.min(bm25_raw), np.max(bm25_raw)
        if bm25_max > bm25_min:
            bm25_norm = (bm25_raw - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_norm = np.zeros_like(bm25_raw)
            
        # 4. Interpolate
        final_scores = (gamma * dense_raw) + ((1.0 - gamma) * bm25_norm)
        
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        return [
            {
                "chunk_id": self.corpus_chunks[idx]['chunk_id'],
                "doc_id": self.corpus_chunks[idx]['doc_id'],
                "text": self.corpus_chunks[idx]['text'],
                "final_score": float(final_scores[idx]),
            }
            for idx in top_indices
        ]

    def retrieve_rrf(self, query: str, top_k: int = 100, k_constant: int = 600, gamma=None, **kwargs) -> list:
        """
        Executes Reciprocal Rank Fusion (RRF) combining BM25 and BGE-m3 ranks.
        Standard IR literature uses k=60.
        """
        # 1. Get Raw Scores
        tokenized_query = self.dictionary.clean_and_tokenize(query, self.language)
        bm25_raw = self.bm25.get_scores(tokenized_query)
        
        query_emb = self.dense_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        dense_raw = torch.matmul(self.dense_embeddings, query_emb).cpu().numpy()
        
        # 2. Convert Scores to Ranks (Fast Numpy Array Ops)
        # argsort()[::-1] gives indices in descending order of score
        bm25_ranked_indices = np.argsort(bm25_raw)[::-1]
        dense_ranked_indices = np.argsort(dense_raw)[::-1]
        
        # Create mapping of (document_index -> rank position)
        bm25_ranks = np.empty_like(bm25_ranked_indices)
        bm25_ranks[bm25_ranked_indices] = np.arange(1, len(bm25_raw) + 1)
        
        dense_ranks = np.empty_like(dense_ranked_indices)
        dense_ranks[dense_ranked_indices] = np.arange(1, len(dense_raw) + 1)
        
        # 3. Apply RRF Formula
        rrf_scores = (1.0 / (k_constant + bm25_ranks)) + (1.0 / (k_constant + dense_ranks))
        
        # 4. Extract Top K
        top_indices = np.argsort(rrf_scores)[::-1][:top_k]
        
        return [
            {
                "chunk_id": self.corpus_chunks[idx]['chunk_id'],
                "doc_id": self.corpus_chunks[idx]['doc_id'],
                "text": self.corpus_chunks[idx]['text'],
                "final_score": float(rrf_scores[idx]),
            }
            for idx in top_indices
        ]

def main(args):
    lang = _DATASET_LANG.get(args.dataset_name, 'en')
    
    loader = UniversalRAGLoader(
        chunk_size=args.chunk_size,
        chunk_size_zh=args.chunk_size * 2,
        chunk_size_ja=args.chunk_size * 2,
        overlap=max(1, args.chunk_size // 7),
        min_df=2
    )
    
    # Load Data (Matching main_pipeline1.py logic)
    loaders = {
        'english_aviation': loader.load_english_aviation,
        'chinese_aviation': loader.load_chinese_aviation,
        'thucnews':         loader.load_thucnews,
        'sogou':            loader.load_sogou,
        'miracl_zh':        loader.load_miracl_zh,
        'trec_covid':       loader.load_beir,
        'nfcorpus':         loader.load_beir,
        'mr_tydi_ja':       lambda fp: loader.load_mr_tydi(fp, 'ja'),
        'mr_tydi_th':       lambda fp: loader.load_mr_tydi(fp, 'th'),
        'medweb_en':        lambda fp: loader.load_medweb(fp, 'en'),
        'medweb_ja':        lambda fp: loader.load_medweb(fp, 'ja'),
        'medweb_zh':        lambda fp: loader.load_medweb(fp, 'zh'),
    }
    df = loaders[args.dataset_name](args.filepath)
    
    # Apply same corpus capping if used in main evaluation
    if args.max_docs and len(df) > args.max_docs:
        df = df.sample(n=args.max_docs, random_state=42).reset_index(drop=True)
        
    all_chunks_dicts, _, _ = loader.process_corpus(df, language=lang)
    
    # Init Router
    router = BM25HybridRouter(dense_model_name='BAAI/bge-m3', dictionary=loader, language=lang)
    
    # Use the same cache path naming convention so it loads the BGE-m3 embeddings you already generated
    cache_path = f"experiment_logs1/cache_{args.dataset_name}_tau0.5_K50"
    router.index_corpus(all_chunks_dicts, cache_path=cache_path)
    
    # Build Evaluation Dataset
    if args.dataset_name in _QRELS_DATASETS:
        queries, qrels = load_queries_and_qrels(args.queries_path, args.qrels_path)
        id_prefix = _QRELS_DATASETS[args.dataset_name]
        eval_dataset = build_evaluation_dataset_qrels(queries, qrels, all_chunks_dicts, id_prefix)
    else:
        eval_dataset = build_evaluation_dataset(df, all_chunks_dicts, args.dataset_name)
        
    # Evaluate baselines
    print("\n" + "="*50)
    print(f"BASELINE EVALUATION: {args.dataset_name.upper()}")
    print("="*50)
    
    # 0.0 = Pure BM25, 0.5 = BM25+Dense Hybrid, 1.0 = Pure Dense
    for weight in [0.0, 0.5, 1.0]:
        label = "Pure BM25" if weight == 0.0 else "Pure Dense" if weight == 1.0 else "BM25+Dense Hybrid"
        print(f"\n--- Testing: {label} (Gamma={weight}) ---")
        evaluate_retrieval(router, eval_dataset, gamma=weight)

    print("\n--- Testing: Reciprocal Rank Fusion (RRF) ---")
    
    # Temporarily override the router's retrieve method with our RRF version
    #original_retrieve = router.retrieve
    router.retrieve = router.retrieve_rrf 
    
    evaluate_retrieval(router, eval_dataset, gamma=None) # Gamma is ignored in RRF
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 and Hybrid Baseline Evaluator")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--queries_path", type=str, default=None)
    parser.add_argument("--qrels_path", type=str, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=150)
    
    args = parser.parse_args()
    main(args)