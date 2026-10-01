#!/usr/bin/env python3
"""
camera_ready_extras.py -- one script for every number of the camera-ready run
of the AACL-IJCNLP 2026 paper, written so that each figure in the paper traces
to one run.

Subcommands
-----------
  medweb        MedWeb EN/JA/ZH: Pure Dense, Pure BM25, RRF (k=60), Lex-TM
                (tau=0.8, gamma=0.7, adaptive gamma on/off), LDA routing
                (tau=1.1, HHI off), topic-only, gamma sweep, BGE-M3 learned-sparse
                and multi-vector (late-interaction) baselines, paired bootstrap
                and randomisation tests, per-query-type breakdown, the Figure 1
                case, and a label-free control (label string removed from the
                indexed documents).
  beir          Any BEIR-format collection (FiQA-2018, CQADupStack, ChatDoctor,
                CmedqaRetrieval export, ...): Dense, BM25, RRF, Lex-TM, LDA
                routing, BGE-M3 sparse (+ multi-vector re-scoring of the union
                of the top-100 dense and top-100 sparse candidates), pair-level
                MRR@20 (pipeline protocol) and standard query-level
                MRR@10 / nDCG@10.
  export_cmedqa Writes C-MTEB/CmedqaRetrieval as BEIR files with a seeded
                10,000-document subsample that keeps every gold document;
                --n_docs 0 writes the full corpus (used in the paper).
  macros        Rebuilds results_generated.tex from the JSON files in --out.
  selftest      Offline test with mock encoders (no GPU, no downloads).

Run it from the directory that holds lex_tm_model.py and rag_dataload.py
(pipelines/v3_patched), or set LEXTM_DIR. Outputs go to --out:
  <out>/medweb_results.json, <out>/beir_<name>_results.json,
  <out>/per_query/*.json, <out>/results_generated.tex  (\\input by the paper)

Scoring replicates rag_evaluation_pipeline_patched.py exactly:
  dense = (cos + 1)/2 ; topic = 1 - scipy.jensenshannon(theta_d, theta_q)
  final = g*dense + (1-g)*topic, g adapted by query-topic entropy when on;
  RR = 1/rank of the first relevant chunk within the top 20, else 0.
A cross-check against LexTMRouter.retrieve() runs automatically when the
pipeline module can be imported (MedWeb, BGE-m3, final configuration).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# ----------------------------------------------------------------------------
# Locate the Lex-TM pipeline modules
# ----------------------------------------------------------------------------
def _find_lextm_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get("LEXTM_DIR"), os.getcwd(), here):
        if cand and os.path.exists(os.path.join(cand, "lex_tm_model.py")):
            cand = os.path.abspath(cand)
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    return None


ENCODERS = {
    # key: (hf id, query prompt name, dtype)
    "bge-m3": ("BAAI/bge-m3", None, None),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", "query", None),
    "qwen3-4b": ("Qwen/Qwen3-Embedding-4B", "query", "float16"),
    "qwen3-8b": ("Qwen/Qwen3-Embedding-8B", "query", "float16"),
}
TOP_K = 20
RRF_K = 60
B_BOOT = 10000
SEED = 42
M3_WEIGHTS = (0.4, 0.2, 0.4)   # dense, sparse, multi-vector (BGE-M3 model card)
MEDWEB_LABELS = ["Influenza", "Diarrhea", "Hayfever", "Cough", "Headache",
                 "Fever", "Runnynose", "Cold"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def require_punkt():
    """English numbers depend on NLTK punkt: without it rag_dataload silently
    falls back to a regex tokenizer (EN vocabulary 921 instead of 909) and every
    English result changes. Fail fast instead."""
    try:
        from nltk.tokenize import word_tokenize
        word_tokenize("A quick check.")
    except Exception as ex:
        raise SystemExit(
            "NLTK punkt/punkt_tab is not available (" + type(ex).__name__ + "). English tokenization "
            "would fall back to a regex tokenizer and change the English results.\n"
            "Fix: python -m nltk.downloader punkt punkt_tab stopwords wordnet")


def model_tag(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


# ----------------------------------------------------------------------------
# Dense encoders
# ----------------------------------------------------------------------------
class DenseEncoder:
    def __init__(self, key, batch_size=64):
        from sentence_transformers import SentenceTransformer
        hf_id, prompt, dtype = ENCODERS[key]
        mk = {"use_safetensors": True}
        if dtype:
            mk["torch_dtype"] = dtype
        log(f"loading dense encoder {hf_id}")
        self.model = SentenceTransformer(hf_id, model_kwargs=mk)
        self.key, self.hf_id, self.prompt, self.bs = key, hf_id, prompt, batch_size

    def encode_docs(self, texts):
        e = self.model.encode(texts, batch_size=self.bs, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(e, dtype=np.float32)

    def encode_queries(self, texts):
        kw = dict(batch_size=self.bs, convert_to_numpy=True, normalize_embeddings=True)
        if self.prompt:
            kw["prompt_name"] = self.prompt
        return np.asarray(self.model.encode(texts, **kw), dtype=np.float32)


def doc_embeddings(enc, texts, cache_file):
    """Load the pipeline's cache when it matches, else encode and save."""
    if cache_file and os.path.exists(cache_file):
        E = np.load(cache_file).astype(np.float32)
        if E.shape[0] == len(texts):
            idx = np.random.default_rng(0).choice(len(texts), size=min(3, len(texts)), replace=False)
            fresh = enc.encode_docs([texts[i] for i in idx])
            cos = np.sum(fresh * E[idx], axis=1)
            if np.all(cos > 0.999):
                log(f"  cached embeddings OK ({cache_file}, spot-check cos={cos.min():.5f})")
                return E
            log(f"  cache spot-check failed (cos={cos.round(4)}); re-encoding")
        else:
            log(f"  cache shape {E.shape} != {len(texts)} docs; re-encoding")
    E = enc.encode_docs(texts)
    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        np.save(cache_file, E)
        log(f"  embeddings cached -> {cache_file}")
    return E


class M3Encoder:
    """BGE-M3 dense, learned-sparse (lexical weights) and multi-vector outputs.
    Uses FlagEmbedding when it is installed (the reference implementation);
    otherwise reproduces its three heads with transformers plus the
    checkpoint's sparse_linear.pt / colbert_linear.pt (same maths)."""

    def __init__(self, batch_size=32, max_length=512):
        self.bs, self.ml = batch_size, max_length
        try:
            from FlagEmbedding import BGEM3FlagModel
            log("loading BGE-M3 via FlagEmbedding (dense + sparse + multi-vector)")
            self.m = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            self.impl = "FlagEmbedding"
        except Exception as ex:
            log(f"  FlagEmbedding not usable ({type(ex).__name__}); using built-in BGE-M3 heads")
            self._init_heads()
            self.impl = "builtin"

    def _init_heads(self):
        import torch
        from transformers import AutoTokenizer, AutoModel
        from huggingface_hub import hf_hub_download
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        enc = AutoModel.from_pretrained("BAAI/bge-m3", use_safetensors=True)
        H = enc.config.hidden_size
        self.sl, self.cl = torch.nn.Linear(H, 1), torch.nn.Linear(H, H)
        self.sl.load_state_dict(torch.load(hf_hub_download("BAAI/bge-m3", "sparse_linear.pt"), map_location="cpu"))
        self.cl.load_state_dict(torch.load(hf_hub_download("BAAI/bge-m3", "colbert_linear.pt"), map_location="cpu"))
        dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.enc = enc.to(self.dev, dtype=dt).eval()
        self.sl = self.sl.to(self.dev, dtype=dt).eval()
        self.cl = self.cl.to(self.dev, dtype=dt).eval()
        self.skip = {int(i) for i in (self.tok.cls_token_id, self.tok.eos_token_id,
                                      self.tok.pad_token_id, self.tok.unk_token_id) if i is not None}

    def encode(self, texts, colbert=True):
        if self.impl == "FlagEmbedding":
            out = self.m.encode(texts, batch_size=self.bs, max_length=self.ml, return_dense=True,
                                return_sparse=True, return_colbert_vecs=colbert)
            dense = np.asarray(out["dense_vecs"], dtype=np.float32)
            sparse = [{str(k): float(v) for k, v in d.items()} for d in out["lexical_weights"]]
            cols = [np.asarray(c, dtype=np.float16) for c in out["colbert_vecs"]] if colbert else None
            return dense, sparse, cols
        torch = self.torch
        dense, sparse, cols = [], [], []
        with torch.no_grad():
            for s0 in range(0, len(texts), self.bs):
                b = self.tok(texts[s0:s0 + self.bs], padding=True, truncation=True,
                             max_length=self.ml, return_tensors="pt").to(self.dev)
                h = self.enc(**b).last_hidden_state
                dense.append(torch.nn.functional.normalize(h[:, 0].float(), dim=-1).cpu().numpy())
                tw = torch.relu(self.sl(h)).squeeze(-1).float().cpu().numpy()
                ids = b["input_ids"].cpu().numpy()
                am = b["attention_mask"]
                for i in range(len(ids)):
                    d = {}
                    for t, w in zip(ids[i], tw[i]):
                        if int(t) in self.skip or w <= 0:
                            continue
                        k = str(int(t))
                        if w > d.get(k, 0.0):
                            d[k] = float(w)
                    sparse.append(d)
                if colbert:
                    cv = self.cl(h[:, 1:]) * am[:, 1:, None].to(h.dtype)
                    cv = torch.nn.functional.normalize(cv.float(), dim=-1).cpu().numpy().astype(np.float16)
                    nt = am.sum(dim=1).cpu().numpy()
                    for i in range(len(ids)):
                        cols.append(cv[i, : int(nt[i]) - 1])
        return np.concatenate(dense), sparse, (cols if colbert else None)


def lexical_matrix(dicts, vocab=None):
    """BGE-M3 lexical weights -> CSR; q.d = compute_lexical_matching_score."""
    from scipy.sparse import csr_matrix
    grow = vocab is None
    vocab = {} if grow else vocab
    rows, cols, vals = [], [], []
    for i, d in enumerate(dicts):
        for t, w in d.items():
            j = vocab.setdefault(t, len(vocab)) if grow else vocab.get(t)
            if j is not None:
                rows.append(i); cols.append(j); vals.append(w)
    return csr_matrix((vals, (rows, cols)), shape=(len(dicts), len(vocab))), vocab


def colbert_scores(qv, dvs, cand, block=256):
    """FlagEmbedding.colbert_score for the candidate docs: mean over query
    tokens of the max dot product; -inf for non-candidates. GPU if present."""
    s = np.full(len(dvs), -np.inf)
    cand = list(cand)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch, dev = None, "cpu"
    q = qv.astype(np.float32)
    for b0 in range(0, len(cand), block):
        cb = cand[b0:b0 + block]
        L = max(len(dvs[j]) for j in cb)
        D = np.zeros((len(cb), L, q.shape[1]), dtype=np.float32)
        M = np.zeros((len(cb), L), dtype=bool)
        for r, j in enumerate(cb):
            D[r, :len(dvs[j])] = dvs[j]
            M[r, :len(dvs[j])] = True
        if torch is not None:
            Dt, Mt, Qt = torch.from_numpy(D).to(dev), torch.from_numpy(M).to(dev), torch.from_numpy(q).to(dev)
            ts = torch.einsum("qh,clh->cql", Qt, Dt).masked_fill(~Mt[:, None, :], -1e4)
            sc = (ts.max(dim=2).values.sum(dim=1) / q.shape[0]).cpu().numpy()
        else:
            ts = np.einsum("qh,clh->cql", q, D)
            ts[~np.broadcast_to(M[:, None, :], ts.shape)] = -1e4
            sc = ts.max(axis=2).sum(axis=1) / q.shape[0]
        s[cb] = sc
    return s


# SPLADE++ checkpoints only (the paper labels the row SPLADE++)
SPLADE_CANDIDATES = ["naver/splade-cocondenser-ensembledistil", "prithivida/Splade_PP_en_v1"]


class SpladeEncoder:
    """SPLADE-max (log(1+ReLU(MLM logits)), max-pooled) with transformers."""

    def __init__(self, names=SPLADE_CANDIDATES, max_length=512, batch_size=16):
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.torch, self.ml, self.bs = torch, max_length, batch_size
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        errs = []
        for nm in names:
            for kw in ({"use_safetensors": True}, {}):
                try:
                    self.tok = AutoTokenizer.from_pretrained(nm)
                    self.m = AutoModelForMaskedLM.from_pretrained(nm, **kw).to(self.dev).eval()
                    self.name = nm
                    log(f"loaded SPLADE checkpoint {nm}")
                    return
                except Exception as ex:
                    errs.append(f"{nm} {kw}: {type(ex).__name__}: {str(ex)[:120]}")
        raise RuntimeError("no SPLADE checkpoint could be loaded: " + " | ".join(errs))

    def encode(self, texts):
        from scipy.sparse import csr_matrix, vstack
        torch = self.torch
        rows = []
        with torch.no_grad():
            for s0 in range(0, len(texts), self.bs):
                b = self.tok(texts[s0:s0 + self.bs], padding=True, truncation=True,
                             max_length=self.ml, return_tensors="pt").to(self.dev)
                with torch.autocast(device_type="cuda", enabled=(self.dev == "cuda")):
                    lg = self.m(**b).logits
                rep = torch.max(torch.log1p(torch.relu(lg.float())) * b["attention_mask"][..., None], dim=1).values
                rows.append(csr_matrix(rep.cpu().numpy()))
        return vstack(rows).tocsr()


class FastBM25:
    """rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25) with identical
    arithmetic, on a sparse term-document matrix so large corpora are fast."""

    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        from scipy.sparse import csc_matrix
        self.k1, self.b = k1, b
        self.vocab, rows, cols, vals = {}, [], [], []
        self.doc_len = np.array([len(d) for d in corpus], dtype=np.float64)
        self.n = len(corpus)
        self.avgdl = float(self.doc_len.sum()) / self.n
        for i, doc in enumerate(corpus):
            for w, c in Counter(doc).items():
                j = self.vocab.setdefault(w, len(self.vocab))
                rows.append(i); cols.append(j); vals.append(c)
        self.tf = csc_matrix((np.array(vals, dtype=np.float64), (rows, cols)), shape=(self.n, len(self.vocab)))
        df = np.diff(self.tf.indptr).astype(np.float64)          # docs per term (csc)
        idf = np.log(self.n - df + 0.5) - np.log(df + 0.5)
        eps = epsilon * (idf.sum() / len(idf))
        self.idf = np.where(idf < 0, eps, idf)
        self.norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

    def get_scores(self, query):
        score = np.zeros(self.n)
        for q in query:                       # duplicates count, as in rank_bm25
            j = self.vocab.get(q)
            if j is None or self.idf[j] == 0:
                continue
            a, z = self.tf.indptr[j], self.tf.indptr[j + 1]
            rows, f = self.tf.indices[a:z], self.tf.data[a:z]
            term = np.zeros(self.n)
            term[rows] = self.idf[j] * (f * (self.k1 + 1) / (f + self.norm[rows]))
            score += term
        return score


# ----------------------------------------------------------------------------
# Lex-TM (topic model + routing), replicating the pipeline exactly
# ----------------------------------------------------------------------------
def train_lextm(bow, id2token, doc_assignments, tau, K, seed=SEED):
    from lex_tm_model import LexTMLdaModel
    m = LexTMLdaModel(num_topics=K, alpha=0.1, beta_base=0.01, tau=tau,
                      lambda_amp=2.0, random_state=seed)
    m.fit(corpus=bow, dictionary=id2token, max_iter=500, convergence_threshold=1e-4,
          min_iter=50, doc_assignments=doc_assignments, hhi_scope="document")
    return m


def get_bow(loader, text, lang):
    counts = {}
    for tok in loader.clean_and_tokenize(text, lang):
        wid = loader.token2id.get(tok)
        if wid is not None:
            counts[wid] = counts.get(wid, 0) + 1
    return list(counts.items())


def topic_matrix(model, loader, texts, lang):
    return np.array([model.get_document_topic_distribution(get_bow(loader, t, lang))
                     for t in texts])


def lextm_scores(dense01, Theta, theta_q, gamma, adaptive):
    from scipy.spatial.distance import jensenshannon
    g = gamma
    if adaptive and g < 1.0:
        K = len(theta_q)
        H = float(-np.sum(theta_q * np.log(theta_q + 1e-10)))
        conf = 1.0 - H / float(np.log(K))
        g = float(np.clip(g - 0.2 * conf, g - 0.2, g))
    topic = 1.0 - jensenshannon(Theta, theta_q[None, :], axis=1)
    return g * dense01 + (1.0 - g) * topic, g


def ranks_desc(scores):
    order = np.argsort(scores)[::-1]
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return order, r


def rr_at(order, ids, gt, k=TOP_K):
    for i, j in enumerate(order[:k]):
        if ids[j] in gt:
            return 1.0 / (i + 1)
    return 0.0


def recall_at(order, ids, gt, k):
    return 1.0 if any(ids[j] in gt for j in order[:k]) else 0.0


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def boot_ci(x, B=B_BOOT, seed=SEED):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b, B=B_BOOT, seed=SEED):
    """Paired bootstrap CI on mean(a-b) (queries resampled jointly) and a
    two-sided paired randomisation (sign-flip) test."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    bm = d[idx].mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(B, len(d)))
    perm = (signs * d).mean(axis=1)
    p_rand = float((np.sum(np.abs(perm) >= abs(d.mean()) - 1e-12) + 1) / (B + 1))
    return dict(delta=float(d.mean()), lo=float(np.percentile(bm, 2.5)),
                hi=float(np.percentile(bm, 97.5)), p=p_rand,
                wins=int(np.sum(d > 0)), losses=int(np.sum(d < 0)), ties=int(np.sum(d == 0)))


def ndcg_at(order, ids, rel, k=10):
    dcg = 0.0
    for i, j in enumerate(order[:k]):
        g = rel.get(ids[j], 0)
        if g:
            dcg += (2 ** g - 1) / math.log2(i + 2)
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ----------------------------------------------------------------------------
# Data helpers (mirror main_pipeline1_patched.py without importing torch)
# ----------------------------------------------------------------------------
def make_loader(chunk_size=512):
    from rag_dataload import UniversalRAGLoader
    return UniversalRAGLoader(chunk_size=chunk_size, chunk_size_zh=chunk_size,
                              chunk_size_ja=chunk_size, overlap=max(1, chunk_size // 7),
                              min_df=2)


def title_groups(df, chunks):
    lookup = defaultdict(set)
    for c in chunks:
        lookup[c["doc_id"]].add(c["chunk_id"])
    groups, rep = {}, {}
    for _, r in df.iterrows():
        q = str(r["ReviewTitle"]).strip()
        if q and r["document_id"] in lookup:
            groups.setdefault(q, set()).update(lookup[r["document_id"]])
            rep.setdefault(q, r["document_id"])
    return [{"query": q, "gt": g} for q, g in groups.items()]


def read_beir(data_dir, split, prefix):
    corpus, queries, qrels = [], {}, defaultdict(dict)
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                corpus.append({"document_id": f"{prefix}{d['_id']}",
                               "full_text": (d.get("title", "") + " " + d["text"]).strip()})
    with open(os.path.join(data_dir, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                queries[str(d["_id"])] = d["text"]
    with open(os.path.join(data_dir, "qrels", f"{split}.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                s = int(float(p[2]))
            except ValueError:
                continue                       # header
            if s > 0:
                qrels[p[0]][f"{prefix}{p[1]}"] = s
    return corpus, queries, qrels


# ----------------------------------------------------------------------------
# Core evaluation over a fixed candidate set
# ----------------------------------------------------------------------------
def run_systems(tag, lang, loader, chunks, bow, doc_assign, queries, gts, encs,
                args, qrels_graded=None, label_free=False, want_crosscheck=False):
    # first occurrence of each distinct query (pair-level items repeat queries)
    seen, uniq = set(), []
    for i, q in enumerate(queries):
        if q not in seen:
            seen.add(q)
            uniq.append(i)
    """queries: list[str]; gts: list[set(chunk_id)]. Returns dict of results."""
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    n = len(ids)
    res = {"n_queries": len(queries), "n_chunks": n, "systems": {}, "per_query": {}}

    # --- topic models (encoder independent) -------------------------------
    t0 = time.time()
    K = args.num_topics
    tm = {}
    for tau in sorted({args.tau, 1.1} | set(args.tau_sweep)):
        log(f"  [{tag}] training topic model tau={tau} K={K}")
        m = train_lextm(bow, loader.id2token, doc_assign, tau, K)
        tm[tau] = (m, topic_matrix(m, loader, texts, lang))
    res["topic_train_seconds"] = round(time.time() - t0, 1)

    # --- BM25 + tie diagnostics -------------------------------------------
    bm = FastBM25([loader.clean_and_tokenize(t, lang) for t in texts])
    bm25_s = [bm.get_scores(loader.clean_and_tokenize(q, lang)) for q in queries]

    def record(name, orders):
        rr = [rr_at(o, ids, g) for o, g in zip(orders, gts)]
        out = {"mrr": float(np.mean(rr)), "ci": boot_ci(rr)[1:],
               "r5": float(np.mean([recall_at(o, ids, g, 5) for o, g in zip(orders, gts)])),
               "r10": float(np.mean([recall_at(o, ids, g, 10) for o, g in zip(orders, gts)])),
               "r20": float(np.mean([recall_at(o, ids, g, 20) for o, g in zip(orders, gts)]))}
        if qrels_graded is not None:
            nd_q = [float(ndcg_at(orders[i], ids, qrels_graded[i])) for i in uniq]
            out["ndcg10"] = float(np.mean(nd_q))
            res.setdefault("per_query_ndcg10", {})[name] = nd_q
            out["mrr10_q"] = float(np.mean([rr_at(orders[i], ids, set(qrels_graded[i]), 10) for i in uniq]))
            out["n_unique_queries"] = len(uniq)
        res["systems"][name] = out
        res["per_query"][name] = rr
        log(f"    {name:<34} MRR={out['mrr']:.4f}  [{out['ci'][0]:.4f}, {out['ci'][1]:.4f}]"
            + (f"  nDCG@10={out['ndcg10']:.4f}" if 'ndcg10' in out else ""))
        return rr

    bm25_orders = [np.argsort(s)[::-1][:TOP_K] for s in bm25_s]
    record("bm25", bm25_orders)
    # tie sensitivity of BM25 (optimistic / pessimistic tie resolution)
    for mode in ("best", "worst"):
        orders = []
        for s, g in zip(bm25_s, gts):
            rel = np.array([i in g for i in ids], dtype=float)
            key = rel if mode == "best" else -rel
            orders.append(np.lexsort((-key, -s))[:TOP_K])
        res.setdefault("bm25_ties", {})[mode] = float(np.mean([rr_at(o, ids, g) for o, g in zip(orders, gts)]))

    fig_cases = []
    for ekey in encs:
        enc = encs[ekey]
        log(f"  [{tag}] encoder {ekey}")
        cache = os.path.join(args.cache_dir, f"emb_{tag}{'_labelfree' if label_free else ''}_{model_tag(enc.hf_id)}_chunk{getattr(args, 'chunk_size', 512)}.npy")
        E = doc_embeddings(enc, texts, cache)
        Q = enc.encode_queries(queries)
        dense_raw = Q @ E.T                     # (nq, n) cosine
        dense01 = (dense_raw + 1.0) / 2.0
        pre = f"{ekey}."
        d_orders = [np.argsort(s)[::-1][:TOP_K] for s in dense01]
        record(pre + "dense", d_orders)
        # RRF exactly as HybridRouter.retrieve_rrf (raw dot products for ranks)
        rrf_orders = []
        for i in range(len(queries)):
            _, rd = ranks_desc(dense_raw[i])
            _, rb = ranks_desc(bm25_s[i])
            rrf = 1.0 / (RRF_K + rd) + 1.0 / (RRF_K + rb)
            rrf_orders.append(np.argsort(rrf)[::-1][:TOP_K])
        record(pre + "rrf", rrf_orders)

        def lex_run(name, tau, gamma, adaptive):
            m, Theta = tm[tau]
            orders, gs = [], []
            for i, q in enumerate(queries):
                th = m.get_document_topic_distribution(get_bow(loader, q, lang))
                sc, g = lextm_scores(dense01[i], Theta, th, gamma, adaptive)
                orders.append(np.argsort(sc)[::-1][:TOP_K])
                gs.append(g)
            record(name, orders)
            res["systems"][name]["gamma_eff_mean"] = float(np.mean(gs))
            return orders

        lex_orders = lex_run(pre + "lextm", args.tau, args.gamma, True)
        lex_run(pre + "lextm_noadapt", args.tau, args.gamma, False)
        lex_run(pre + "lda_route", 1.1, args.gamma, False)
        lex_run(pre + "topic_only", args.tau, 0.0, False)
        for g in args.gamma_sweep:
            lex_run(pre + f"lextm_g{g}", args.tau, g, False)
        for t in args.tau_sweep:
            if t not in (args.tau, 1.1):
                lex_run(pre + f"lextm_t{t}", t, args.gamma, False)

        # Gibbs-seed robustness (first encoder only, MRR only)
        seeds = [sd for sd in getattr(args, "seeds", []) if sd != SEED]
        if seeds and not label_free and ekey == list(encs)[0]:
            def mrr_of(m, Theta, gamma, adaptive):
                rr = []
                for i, q in enumerate(queries):
                    th = m.get_document_topic_distribution(get_bow(loader, q, lang))
                    sc, _ = lextm_scores(dense01[i], Theta, th, gamma, adaptive)
                    rr.append(rr_at(np.argsort(sc)[::-1][:TOP_K], ids, gts[i]))
                return float(np.mean(rr))
            runs = {str(SEED): {"lextm": res["systems"][pre + "lextm"]["mrr"],
                                "lextm_noadapt": res["systems"][pre + "lextm_noadapt"]["mrr"],
                                "lda_route": res["systems"][pre + "lda_route"]["mrr"]}}
            try:
                for sd in seeds:
                    log(f"  [{tag}] Gibbs seed {sd}")
                    m8 = train_lextm(bow, loader.id2token, doc_assign, args.tau, K, seed=sd)
                    T8 = topic_matrix(m8, loader, texts, lang)
                    m11 = train_lextm(bow, loader.id2token, doc_assign, 1.1, K, seed=sd)
                    T11 = topic_matrix(m11, loader, texts, lang)
                    runs[str(sd)] = {"lextm": mrr_of(m8, T8, args.gamma, True),
                                     "lextm_noadapt": mrr_of(m8, T8, args.gamma, False),
                                     "lda_route": mrr_of(m11, T11, args.gamma, False)}
                    log(f"    seed {sd}: " + ", ".join(f"{k}={v:.4f}" for k, v in runs[str(sd)].items()))
                res["seed_runs"] = {"encoder": ekey, "runs": runs}
            except Exception as ex:
                log(f"    seed runs skipped: {type(ex).__name__}: {ex}")
                res["seed_runs_error"] = f"{type(ex).__name__}: {ex}"

        # paired tests: Lex-TM (final) vs everything else
        a = res["per_query"][pre + "lextm"]
        tests = {}
        for other in [pre + "rrf", "bm25", pre + "dense", pre + "lda_route", pre + "lextm_noadapt"]:
            tests[other] = paired(a, res["per_query"][other])
        res.setdefault("paired", {})[ekey] = tests
        for o, t in tests.items():
            log(f"    Lex-TM vs {o:<24} delta={t['delta']:+.4f} CI[{t['lo']:+.4f},{t['hi']:+.4f}] p={t['p']:.4f}")

        # Figure 1 candidates (MedWeb only): Lex-TM #1 relevant, Dense #1 not
        if want_crosscheck is not None:
            for i, (q, g) in enumerate(zip(queries, gts)):
                lo, do = lex_orders[i], d_orders[i]
                if ids[lo[0]] in g and ids[do[0]] not in g:
                    full_d = np.argsort(dense01[i])[::-1]
                    dr = next((k + 1 for k, j in enumerate(full_d) if ids[j] in g), 0)
                    fig_cases.append({"encoder": ekey, "query": q, "dense_rank_of_truth": dr,
                                      "dense_top_chunk": ids[do[0]], "dense_top_text": texts[do[0]],
                                      "lextm_top_chunk": ids[lo[0]], "lextm_top_text": texts[lo[0]],
                                      "group_size": len(g)})

        # exact cross-check against the pipeline's own retrieve()
        if want_crosscheck and ekey == "bge-m3":
            res["crosscheck"] = crosscheck(enc, tm[args.tau][0], loader, lang, chunks, E,
                                           tm[args.tau][1], queries, lex_orders, args)
    extra_baselines(res, record, lang, texts, queries, args)
    fig_cases.sort(key=lambda c: (c["dense_rank_of_truth"] == 0, c["dense_rank_of_truth"]), reverse=True)
    res["figure1_candidates"] = fig_cases[:15]
    return res


def extra_baselines(res, record, lang, texts, queries, args):
    """Learned-sparse and late-interaction baselines (encoder independent)."""
    n = len(texts)
    ref = "bge-m3.lextm" if "bge-m3.lextm" in res["per_query"] else None

    def pair_all(names):
        if ref:
            for nm in names:
                res.setdefault("paired", {}).setdefault("bge-m3", {})[nm] = paired(
                    res["per_query"][ref], res["per_query"][nm])

    if args.m3:
        try:
            m3 = M3Encoder()
            res["m3_impl"] = m3.impl
            mv = not getattr(args, "no_multivector", False)
            keep_cols = mv and n <= args.m3_colbert_max_docs
            Dd, Ds, Dc = m3.encode(texts, colbert=keep_cols)
            Qd, Qs, Qc = m3.encode(queries, colbert=mv)
            DS, voc = lexical_matrix(Ds)
            QS, _ = lexical_matrix(Qs, voc)
            SP = (QS @ DS.T).toarray()
            exhaustive = n <= args.m3_exhaustive_max
            res["m3_multivector_scope"] = ("exhaustive" if exhaustive else
                                           "re-scores top-100 dense U top-100 sparse" +
                                           ("" if keep_cols else " (encoded on the fly)"))
            s_orders, c_orders, a_orders = [], [], []
            for i in range(len(queries)):
                sp_i = SP[i]
                s_orders.append(np.argsort(sp_i)[::-1][:TOP_K])
                if not mv:
                    continue
                dn = Qd[i] @ Dd.T
                cand = (list(range(n)) if exhaustive else
                        sorted(set(np.argsort(dn)[::-1][:100]) | set(np.argsort(sp_i)[::-1][:100])))
                if keep_cols:
                    cb = colbert_scores(Qc[i], Dc, cand)
                else:
                    _, _, cc = m3.encode([texts[j] for j in cand], colbert=True)
                    tmp = [None] * n
                    for j, v in zip(cand, cc):
                        tmp[j] = v
                    cb = colbert_scores(Qc[i], tmp, cand)
                c_orders.append(np.argsort(cb)[::-1][:TOP_K])
                w = M3_WEIGHTS
                allsc = np.where(np.isfinite(cb), w[0] * dn + w[1] * sp_i + w[2] * cb, -np.inf)
                a_orders.append(np.argsort(allsc)[::-1][:TOP_K])
                if (i + 1) % 500 == 0:
                    log(f"    BGE-M3 multi-vector: {i + 1}/{len(queries)} queries")
            record("m3.sparse", s_orders)
            done = ["m3.sparse"]
            if mv:
                record("m3.multivector", c_orders)
                record("m3.all", a_orders)
                done += ["m3.multivector", "m3.all"]
            pair_all(done)
        except Exception as ex:
            log(f"    BGE-M3 sparse/multi-vector skipped: {type(ex).__name__}: {ex}")
            res["m3_error"] = f"{type(ex).__name__}: {ex}"
    if args.splade and lang == "en":
        try:
            sp = SpladeEncoder()
            res["splade_model"] = sp.name
            D = sp.encode(texts)
            Q = sp.encode(queries)
            S = (Q @ D.T).toarray()
            record("splade", [np.argsort(S[i])[::-1][:TOP_K] for i in range(len(queries))])
            pair_all(["splade"])
        except Exception as ex:
            log(f"    SPLADE skipped: {type(ex).__name__}: {ex}")
            res["splade_error"] = f"{type(ex).__name__}: {ex}"


def crosscheck(enc, model, loader, lang, chunks, E, Theta, queries, our_orders, args):
    try:
        import torch
        from rag_evaluation_pipeline_patched import LexTMRouter
    except Exception as ex:
        return {"status": f"skipped ({type(ex).__name__})"}
    r = LexTMRouter.__new__(LexTMRouter)
    dev = getattr(enc.model, "device", "cpu")
    r.dense_model, r.dense_model_name = enc.model, enc.hf_id
    r.lex_tm_model, r.dictionary, r.language = model, loader, lang
    r.corpus_texts = [c["text"] for c in chunks]
    r.corpus_chunk_ids = [c["chunk_id"] for c in chunks]
    r.corpus_doc_ids = [c["doc_id"] for c in chunks]
    r.dense_embeddings = torch.tensor(E, dtype=torch.float32).to(dev)
    r.topic_distributions = Theta
    r.min_query_tokens, r.query_smoothing, r.adaptive_gamma = 0, 0.0, True
    r.query_prompt_name, r.embed_batch_size = enc.prompt, 64
    same = 0
    for q, o in zip(queries, our_orders):
        hits = [h["chunk_id"] for h in r.retrieve(q, top_k=TOP_K, gamma=args.gamma)]
        same += int(hits == [r.corpus_chunk_ids[j] for j in o[:TOP_K]])
    st = {"status": "ok", "identical_top20": same, "n": len(queries)}
    log(f"    cross-check vs LexTMRouter.retrieve(): {same}/{len(queries)} identical top-20 lists")
    return st


# ----------------------------------------------------------------------------
# MedWeb
# ----------------------------------------------------------------------------
def n_labels(q):
    if "：" in q or ":" in q:
        body = re.split(r"[:：]", q, maxsplit=1)[1]
        return len([x for x in re.split(r"[,，]", body) if x.strip(" 。.")])
    return 0


def cmd_medweb(args):
    import pandas as pd
    if "en" in args.langs:
        require_punkt()
    encs = {k: DenseEncoder(k, args.batch_size) for k in args.encoders}
    out = {"config": vars(args).copy(), "langs": {}}
    out["config"].pop("func", None)
    for lang in args.langs:
        path = os.path.join(args.medweb_dir, f"medweb_rag_{lang}_fixed.csv")
        log(f"=== MedWeb {lang.upper()} ({path})")
        L = make_loader()
        df = L.load_medweb(path, lang)
        raw = pd.read_csv(path, encoding="utf-8-sig")
        raw["document_id"] = "medweb_" + raw["document_id"].astype(str)
        df = df.merge(raw[["document_id", "ReviewText"]], on="document_id")
        chunks, bow, da = L.process_corpus(df[["document_id", "ReviewTitle", "full_text"]], language=lang)
        ev = title_groups(df, chunks)
        queries, gts = [e["query"] for e in ev], [e["gt"] for e in ev]
        res = run_systems(f"medweb_{lang}", lang, L, chunks, bow, da, queries, gts, encs, args,
                          want_crosscheck=not args.no_crosscheck)
        # query-type breakdown
        # 0-1 labels (the single no-symptom set joins the one-label sets), 2, 3+
        cat = {q: ("01" if n_labels(q) <= 1 else "2" if n_labels(q) == 2 else "3+") for q in queries}
        size = {q: ("large (>=100)" if len(g) >= 100 else "mid (10-99)" if len(g) >= 10 else "small (<10)")
                for q, g in zip(queries, gts)}
        brk = {}
        for name, rr in res["per_query"].items():
            for part, lab in (("labels", cat), ("group", size)):
                acc = defaultdict(list)
                for q, v in zip(queries, rr):
                    acc[lab[q]].append(v)
                brk.setdefault(part, {}).setdefault(name, {k: [float(np.mean(v)), len(v)] for k, v in acc.items()})
        res["breakdown"] = brk
        res["queries"] = queries
        res["group_sizes"] = [len(g) for g in gts]
        # label-free control: index the tweet only, same label-string queries
        if not args.no_label_free:
            log(f"=== MedWeb {lang.upper()} label-free control (tweet text only)")
            L2 = make_loader()
            df2 = df.copy()
            df2["full_text"] = df2["ReviewText"].astype(str)
            ch2, bow2, da2 = L2.process_corpus(df2[["document_id", "ReviewTitle", "full_text"]], language=lang)
            ev2 = title_groups(df2, ch2)
            lf_args = argparse.Namespace(**vars(args))
            lf_args.gamma_sweep, lf_args.tau_sweep, lf_args.seeds = [], [], []
            res["label_free"] = run_systems(f"medweb_{lang}", lang, L2, ch2, bow2, da2,
                                            [e["query"] for e in ev2], [e["gt"] for e in ev2],
                                            encs, lf_args, label_free=True, want_crosscheck=None)
            dump_per_query(args.out, f"medweb_{lang}_label_free.json",
                           {"queries": [e["query"] for e in ev2],
                            "rr_at20": res["label_free"].pop("per_query", {})})
        # length statistics
        if lang == "en" and L.vocab_size != 909:
            log(f"WARNING: English vocabulary is {L.vocab_size}, not 909 as in the logged runs; "
                "check the NLTK installation before using these numbers.")
        res["stats"] = {
            "docs": len(df), "vocab": L.vocab_size,
            "content_tokens_with_label": float(np.mean([len(L.clean_and_tokenize(t, lang)) for t in df["full_text"]])),
            "content_tokens_tweet": float(np.mean([len(L.clean_and_tokenize(str(t), lang)) for t in df["ReviewText"]])),
            "query_content_tokens": float(np.mean([len(L.clean_and_tokenize(q, lang)) for q in queries])),
        }
        out["langs"][lang] = res
        dump(out, args.out, "medweb_results.json")
    write_macros(args.out)
    log("done. Re-compile the paper: results_generated.tex was updated.")


# ----------------------------------------------------------------------------
# BEIR-format collections
# ----------------------------------------------------------------------------
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def cmd_beir(args):
    import pandas as pd
    if args.lang == "en":
        require_punkt()
    if args.download and not os.path.exists(os.path.join(args.data_dir, "corpus.jsonl")):
        import io, urllib.request, zipfile
        top = args.dataset.split("/")[0]
        url = BEIR_URL.format(top)
        norm = os.path.normpath(os.path.abspath(args.data_dir))
        ds = os.path.normpath(args.dataset)
        root = norm[: -len(ds)] if norm.endswith(ds) else os.path.dirname(norm)
        log(f"downloading {url} -> {root}")
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()))
        z.extractall(root)
    prefix = f"{args.dataset.replace('/', '_')}_"
    corpus, queries, qrels = read_beir(args.data_dir, args.split, prefix)
    df = pd.DataFrame(corpus)
    df = df[df["full_text"].str.strip().astype(bool)]
    if args.max_docs and len(df) > args.max_docs:
        keep = set(d for r in qrels.values() for d in r)
        gold = df[df["document_id"].isin(keep)]
        rest = df[~df["document_id"].isin(keep)].sample(n=max(0, args.max_docs - len(gold)), random_state=SEED)
        df = pd.concat([gold, rest]).sort_values("document_id").reset_index(drop=True)
        log(f"  subsampled to {len(df)} docs (all {len(gold)} gold docs kept, seed {SEED})")
    L = make_loader(args.chunk_size)
    chunks, bow, da = L.process_corpus(df[["document_id", "full_text"]], language=args.lang)
    by_doc = defaultdict(set)
    for c in chunks:
        by_doc[c["doc_id"]].add(c["chunk_id"])
    # pair-level items (pipeline protocol) and query-level graded qrels
    qs, gts, graded = [], [], []
    for qid, rel in qrels.items():
        if qid not in queries:
            continue
        for did in rel:
            if did in by_doc:
                qs.append(queries[qid])
                gts.append(by_doc[did])
                graded.append({cid: rel[d] for d in rel if d in by_doc for cid in by_doc[d]})
    log(f"  {len(qs)} (query, relevant doc) pairs over {len(set(qs))} queries, {len(chunks)} chunks")
    if args.max_queries and len(set(qs)) > args.max_queries:
        rng = np.random.default_rng(SEED)
        keepq = set(rng.choice(sorted(set(qs)), size=args.max_queries, replace=False))
        sel = [i for i, q in enumerate(qs) if q in keepq]
        qs, gts, graded = [qs[i] for i in sel], [gts[i] for i in sel], [graded[i] for i in sel]
        log(f"  query subsample: {args.max_queries} queries (seed {SEED})")
    encs = {k: DenseEncoder(k, args.batch_size) for k in args.encoders}
    res = run_systems(prefix.rstrip("_"), args.lang, L, chunks, bow, da, qs, gts, encs, args,
                      qrels_graded=graded, want_crosscheck=None)
    dump_per_query(args.out, f"beir_{prefix.rstrip('_')}.json",
                   {"pairs": [{"query": q, "relevant_chunks": sorted(g)} for q, g in zip(qs, gts)],
                    "rr_at20_pairs": res.pop("per_query", {}),
                    "unique_queries": list(dict.fromkeys(qs)),
                    "ndcg10_per_query": res.pop("per_query_ndcg10", {})})
    res["stats"] = {"docs": len(df), "queries": len(set(qs)), "pairs": len(qs),
                    "content_tokens_doc": float(np.mean([len(c["clean_tokens"]) for c in chunks])),
                    "content_tokens_query": float(np.mean([len(L.clean_and_tokenize(q, args.lang)) for q in set(qs)]))}
    dump({"config": {k: v for k, v in vars(args).items() if k != "func"}, "result": res},
         args.out, f"beir_{prefix.rstrip('_')}_results.json")
    write_macros(args.out)


# ----------------------------------------------------------------------------
# CmedqaRetrieval export (C-MTEB) with a seeded 10k subsample
# ----------------------------------------------------------------------------
def cmd_export_cmedqa(args):
    from datasets import load_dataset
    os.makedirs(os.path.join(args.out_dir, "qrels"), exist_ok=True)
    corpus = queries = qrels = None
    errs = []
    for loader_fn in (
        lambda: (load_dataset("C-MTEB/CmedqaRetrieval", split="corpus"),
                 load_dataset("C-MTEB/CmedqaRetrieval", split="queries"),
                 load_dataset("C-MTEB/CmedqaRetrieval-qrels", split="dev")),
        lambda: (load_dataset("mteb/CmedqaRetrieval", "corpus", split="dev"),
                 load_dataset("mteb/CmedqaRetrieval", "queries", split="dev"),
                 load_dataset("mteb/CmedqaRetrieval", "qrels", split="dev")),
    ):
        try:
            corpus, queries, qrels = loader_fn()
            break
        except Exception as ex:
            errs.append(f"{type(ex).__name__}: {ex}")
    if corpus is None:
        sys.exit("could not load CmedqaRetrieval:\n  " + "\n  ".join(errs))
    # normalise column names across the two hub layouts
    def col(row, *names):
        for nm in names:
            if nm in row:
                return row[nm]
        raise KeyError(names)
    qrels = [{"qid": str(col(r, "qid", "query-id")), "pid": str(col(r, "pid", "corpus-id")),
              "score": int(col(r, "score"))} for r in qrels]
    corpus = [{"id": str(col(r, "id", "_id")), "text": col(r, "text")} for r in corpus]
    queries = [{"id": str(col(r, "id", "_id")), "text": col(r, "text")} for r in queries]
    gold = {str(r["pid"]) for r in qrels}
    ids = [str(r["id"]) for r in corpus]
    rng = np.random.default_rng(SEED)
    rest = [i for i in ids if i not in gold]
    fill = set(rng.choice(rest, size=max(0, args.n_docs - len(gold)), replace=False)) if args.n_docs else set(rest)
    keep = gold | fill
    with open(os.path.join(args.out_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for r in corpus:
            if str(r["id"]) in keep:
                f.write(json.dumps({"_id": str(r["id"]), "title": "", "text": r["text"]}, ensure_ascii=False) + "\n")
    qids = {str(r["qid"]) for r in qrels}
    with open(os.path.join(args.out_dir, "queries.jsonl"), "w", encoding="utf-8") as f:
        for r in queries:
            if str(r["id"]) in qids:
                f.write(json.dumps({"_id": str(r["id"]), "text": r["text"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "qrels", "dev.tsv"), "w", encoding="utf-8") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for r in qrels:
            f.write(f"{r['qid']}\t{r['pid']}\t{int(r['score'])}\n")
    log(f"wrote {len(keep)} docs ({len(gold)} gold), {len(qids)} queries -> {args.out_dir}")


# ----------------------------------------------------------------------------
# LaTeX macros consumed by the paper (\R{key})
# ----------------------------------------------------------------------------
def dump(obj, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    log(f"  wrote {p}")


def dump_per_query(out, name, obj):
    """Per-query values that do not fit in the main results file."""
    d = os.path.join(out, "per_query")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    log(f"  wrote {p}")


def _f(x, nd=4):
    return f"{x:.{nd}f}"


def _sg(x, nd=3):
    """signed number with a typographic minus for LaTeX"""
    return f"{x:+.{nd}f}".replace("-", "$-$")


def _ascii_text(s):
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
          .replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-").replace("\u2026", "..."))
    return "".join(ch for ch in s if ord(ch) < 128)


def _tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def write_macros(out):
    lines = ["% generated by camera_ready_extras.py -- do not edit by hand",
             f"% {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    def put(key, val):
        lines.append(f"\\expandafter\\def\\csname r@{key}\\endcsname{{{val}}}")

    mp = os.path.join(out, "medweb_results.json")
    if os.path.exists(mp):
        mw = json.load(open(mp, encoding="utf-8"))
        cfg = mw.get("config", {})
        agg = {}
        put("mw.cfg.tau", str(cfg.get("tau", "")))
        put("mw.cfg.gamma", str(cfg.get("gamma", "")))
        put("mw.gsweep.tau", str(cfg.get("tau", "")))
        put("mw.cfg.ciB", f"{B_BOOT:,}")
        for lang, r in mw["langs"].items():
            for sysname, s in r["systems"].items():
                k = f"mw.{lang}.{sysname}"
                put(k, _f(s["mrr"]))
                put(k + ".lo", _f(s["ci"][0], 2))
                put(k + ".hi", _f(s["ci"][1], 2))
            for ekey, tests in r.get("paired", {}).items():
                for other, t in tests.items():
                    k = f"mw.{lang}.pair.{ekey}.{other}"
                    put(k + ".d", _sg(t["delta"]))
                    put(k + ".abs", f"{abs(t['delta']):.3f}")
                    put(k + ".lo", _sg(t["lo"]))
                    put(k + ".hi", _sg(t["hi"]))
                    put(k + ".p", f"{t['p']:.3f}" if t["p"] >= 0.001 else "$<$0.001")
                    put(k + ".wl", f"{t['wins']}/{t['losses']}")
            S = {k: v["mrr"] for k, v in r["systems"].items()}
            g_cfg = float(cfg.get("gamma", 0.7))
            if "bge-m3.lextm_noadapt" in S and f"bge-m3.lextm_g{g_cfg}" not in S:
                put(f"mw.{lang}.bge-m3.lextm_g{g_cfg}", _f(S["bge-m3.lextm_noadapt"]))
            tau_vals = [S[k] for k in ("bge-m3.lextm_noadapt", "bge-m3.lda_route") if k in S] + \
                       [v for k, v in S.items() if k.startswith("bge-m3.lextm_t")]
            if len(tau_vals) >= 2:
                agg.setdefault("taurange", {})[lang] = max(tau_vals) - min(tau_vals)
                put(f"mw.{lang}.taurange", _f(max(tau_vals) - min(tau_vals)))
            if "bge-m3.lextm" in S and "bge-m3.lextm_noadapt" in S:
                agg.setdefault("adapt", []).append(abs(S["bge-m3.lextm"] - S["bge-m3.lextm_noadapt"]))
            if "bge-m3.lda_route" in S:
                for k in ("bge-m3.lextm", "bge-m3.lextm_noadapt"):
                    if k in S:
                        agg.setdefault("prior", []).append(abs(S[k] - S["bge-m3.lda_route"]))
            if "bge-m3.rrf" in r["systems"] and "bge-m3.lextm" in r["systems"]:
                a, b = r["systems"]["bge-m3.lextm"]["mrr"], r["systems"]["bge-m3.rrf"]["mrr"]
                put(f"mw.{lang}.rel_rrf", _sg(100 * (a / b - 1), 1))
            if "bm25_ties" in r:
                put(f"mw.{lang}.bm25.best", _f(r["bm25_ties"]["best"]))
                put(f"mw.{lang}.bm25.worst", _f(r["bm25_ties"]["worst"]))
            for part, tab in r.get("breakdown", {}).items():
                for sysname, cells in tab.items():
                    for c, (v, nq) in cells.items():
                        cc = re.sub(r"[^A-Za-z0-9+]", "", c)
                        put(f"mw.{lang}.bk.{part}.{sysname}.{cc}", _f(v, 3))
                        put(f"mw.{lang}.bk.{part}.n.{cc}", str(nq))
            if "label_free" in r:
                for sysname, s in r["label_free"]["systems"].items():
                    put(f"mw.{lang}.lf.{sysname}", _f(s["mrr"]))
                for ekey, tests in r["label_free"].get("paired", {}).items():
                    for other, t in tests.items():
                        k = f"mw.{lang}.lf.pair.{ekey}.{other}"
                        put(k + ".d", _sg(t["delta"]))
                        put(k + ".abs", f"{abs(t['delta']):.3f}")
                        put(k + ".lo", _sg(t["lo"]))
                        put(k + ".hi", _sg(t["hi"]))
                        put(k + ".p", f"{t['p']:.3f}" if t["p"] >= 0.001 else "$<$0.001")
            for k2, v in r.get("stats", {}).items():
                put(f"mw.{lang}.stat.{k2}", _f(v, 1) if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v)))
            cands = [c for c in r.get("figure1_candidates", []) if c["encoder"] == "bge-m3"]
            pick = int(os.environ.get(f"FIG1_{lang.upper()}", "0"))
            if cands:
                c = cands[min(pick, len(cands) - 1)]
                clean = _ascii_text if lang == "en" else str
                put(f"mw.{lang}.fig.query", _tex_escape(clean(c["query"])))
                for side, key in (("dense", "dense_top_text"), ("lex", "lextm_top_text")):
                    if not c.get(key):          # document text not released
                        continue
                    lab, _, tw = c[key].partition(".. ") if lang == "en" else c[key].partition("。.")
                    put(f"mw.{lang}.fig.{side}label", _tex_escape(clean((lab + ("." if lang == "en" else "。")).strip())))
                    tw = clean(tw.strip())
                    put(f"mw.{lang}.fig.{side}tweet", _tex_escape(tw if len(tw) <= 150 else tw[:147].rsplit(" ", 1)[0] + "..."))
                put(f"mw.{lang}.fig.denserank", f"at rank {c['dense_rank_of_truth']}" if c["dense_rank_of_truth"] else "outside the top 20")
                put(f"mw.{lang}.fig.ncands", str(len(cands)))
            for k in ("m3_impl", "splade_model", "m3_multivector_scope"):
                if k in r:
                    put(f"mw.{lang}.{k}", _tex_escape(str(r[k])))
            if "crosscheck" in r:
                put(f"mw.{lang}.crosscheck", f"{r['crosscheck'].get('identical_top20', '-')}/{r['crosscheck'].get('n', '-')}")
            sr = r.get("seed_runs", {}).get("runs", {})
            if len(sr) >= 2:
                put(f"mw.{lang}.seed.n", str(len(sr)))
                for sysname in ("lextm", "lextm_noadapt", "lda_route"):
                    v = np.array([x[sysname] for x in sr.values()])
                    put(f"mw.{lang}.seed.{sysname}.min", _f(v.min()))
                    put(f"mw.{lang}.seed.{sysname}.max", _f(v.max()))
                    put(f"mw.{lang}.seed.{sysname}.mean", _f(v.mean()))
                    put(f"mw.{lang}.seed.{sysname}.sd", _f(v.std(ddof=1)))
                dd = np.array([x["lextm_noadapt"] - x["lda_route"] for x in sr.values()])
                put(f"mw.{lang}.seed.prior.mean", _sg(dd.mean(), 4))
                put(f"mw.{lang}.seed.prior.sd", _f(dd.std(ddof=1)))
                agg.setdefault("seedrange", []).append(float(np.ptp([x["lextm"] for x in sr.values()])))
        if agg.get("adapt"):
            put("mw.adapt.maxdiff", _f(max(agg["adapt"])))
        if agg.get("prior"):
            put("mw.prior.maxdiff", _f(max(agg["prior"])))
        tr = agg.get("taurange", {})
        if "en" in tr and "zh" in tr:
            put("mw.enzh.taurange", _f(max(tr["en"], tr["zh"])))
        if agg.get("seedrange"):
            put("mw.seed.maxrange", _f(max(agg["seedrange"])))
    for fn in sorted(os.listdir(out)) if os.path.isdir(out) else []:
        if fn.startswith("beir_") and fn.endswith("_results.json"):
            b = json.load(open(os.path.join(out, fn), encoding="utf-8"))["result"]
            name = fn[len("beir_"):-len("_results.json")]
            for sysname, s in b["systems"].items():
                k = f"beir.{name}.{sysname}"
                put(k, _f(s["mrr"]))
                if "ndcg10" in s:
                    put(k + ".ndcg", _f(s["ndcg10"]))
                    put(k + ".mrr10", _f(s["mrr10_q"]))
            # Lex-TM minus dense (nDCG@10 when available, else pair-level MRR@20)
            for enc in ("bge-m3", "qwen3-0.6b"):
                a, d = b["systems"].get(f"{enc}.lextm"), b["systems"].get(f"{enc}.dense")
                if a and d:
                    m = "ndcg10" if ("ndcg10" in a and "ndcg10" in d) else "mrr"
                    dv = _sg(a[m] - d[m])
                    put(f"beir.{name}.dlex.{enc}", dv)
                    if enc == "bge-m3":
                        put(f"beir.{name}.dlex", dv)
            for ekey, tests in b.get("paired", {}).items():
                for other, t in tests.items():
                    k = f"beir.{name}.pair.{ekey}.{other}"
                    put(k + ".d", _sg(t["delta"]))
                    put(k + ".lo", _sg(t["lo"]))
                    put(k + ".hi", _sg(t["hi"]))
                    put(k + ".p", f"{t['p']:.3f}" if t["p"] >= 0.001 else "$<$0.001")
            for k2, v in b.get("stats", {}).items():
                put(f"beir.{name}.stat.{k2}", _f(v, 1) if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v)))
    p = os.path.join(out, "results_generated.tex")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"  wrote {p} ({len(lines) - 2} macros)")


def cmd_macros(args):
    write_macros(args.out)


# ----------------------------------------------------------------------------
# Offline self-test with mock encoders (no GPU, no downloads)
# ----------------------------------------------------------------------------
def cmd_selftest(args):
    import types
    import pandas as pd

    def hvec(text, dim=64):
        v = np.zeros(dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    class FakeST:
        def __init__(self, *a, **k):
            self.device = "cpu"

        def encode(self, texts, **kw):
            single = isinstance(texts, str)
            arr = np.stack([hvec(t) for t in ([texts] if single else texts)])
            return arr[0] if single else arr

    class FakeM3:
        def __init__(self, *a, **k):
            pass

        def encode(self, texts, return_colbert_vecs=True, **kw):
            toks = [re.findall(r"\w+", t.lower()) or ["_"] for t in texts]
            return {"dense_vecs": np.stack([hvec(t) for t in texts]),
                    "lexical_weights": [{w: 1.0 / len(ts) for w in ts} for ts in toks],
                    "colbert_vecs": [np.stack([hvec(w) for w in ts]) for ts in toks]}

    sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=FakeST)
    sys.modules["FlagEmbedding"] = types.SimpleNamespace(BGEM3FlagModel=FakeM3)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(240):
        labs = [l for l in MEDWEB_LABELS if rng.random() < 0.18]
        title = ("Patient symptoms: " + ", ".join(labs) + ".") if labs else "Routine health status."
        words = " ".join(rng.choice(["fever", "nose", "cough", "pollen", "sleep", "work", "head",
                                     "stomach", "runny", "tired", "doctor", "medicine"], size=6))
        rows.append({"document_id": f"{i + 1}en", "ReviewTitle": title, "ReviewText": f"I feel {words}"})
    tmp = os.path.join(args.out, "_selftest")
    os.makedirs(tmp, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(tmp, "medweb_rag_en_fixed.csv"), index=False, encoding="utf-8-sig")
    ns = argparse.Namespace(medweb_dir=tmp, out=tmp, langs=["en"], encoders=["bge-m3"],
                            batch_size=16, cache_dir=os.path.join(tmp, "cache"), gamma=0.7, tau=0.8,
                            num_topics=5, gamma_sweep=[0.9], tau_sweep=[0.5], seeds=[42, 43], m3=True, splade=False,
                            m3_exhaustive_max=5000, m3_colbert_max_docs=20000,
                            no_label_free=False, no_crosscheck=True)
    cmd_medweb(ns)
    r = json.load(open(os.path.join(tmp, "medweb_results.json")))["langs"]["en"]
    need = ["bm25", "bge-m3.dense", "bge-m3.rrf", "bge-m3.lextm", "bge-m3.lda_route",
            "m3.sparse", "m3.multivector", "m3.all"]
    missing = [k for k in need if k not in r["systems"]]
    assert not missing, f"missing systems: {missing}"
    assert os.path.exists(os.path.join(tmp, "results_generated.tex"))
    log("SELFTEST PASSED")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out", default="camera_ready_results")
        sp.add_argument("--encoders", default="bge-m3", type=lambda s: s.split(","))
        sp.add_argument("--batch_size", type=int, default=64)
        sp.add_argument("--cache_dir", default="experiment_logs1")
        sp.add_argument("--gamma", type=float, default=0.7)
        sp.add_argument("--tau", type=float, default=0.8)
        sp.add_argument("--no_m3", dest="m3", action="store_false")
        sp.add_argument("--no_splade", dest="splade", action="store_false")
        sp.add_argument("--m3_exhaustive_max", type=int, default=5000,
                        help="multi-vector scored exhaustively up to this many docs, else re-scores candidates")
        sp.add_argument("--m3_colbert_max_docs", type=int, default=20000,
                        help="above this corpus size multi-vector doc vectors are encoded on the fly")
        sp.add_argument("--no_multivector", action="store_true",
                        help="BGE-M3 sparse only (skip the multi-vector and combined modes)")

    sp = sub.add_parser("medweb")
    common(sp)
    sp.add_argument("--medweb_dir", required=True)
    sp.add_argument("--langs", default="en,ja,zh", type=lambda s: s.split(","))
    sp.add_argument("--num_topics", type=int, default=20)
    sp.add_argument("--gamma_sweep", default="0.5,0.6,0.8,0.9", type=lambda s: [float(x) for x in s.split(",") if x])
    sp.add_argument("--tau_sweep", default="0.5", type=lambda s: [float(x) for x in s.split(",") if x])
    sp.add_argument("--seeds", default="42,43,44,45,46", type=lambda s: [int(x) for x in s.split(",") if x],
                    help="Gibbs seeds for the robustness check (42 is the main run)")
    sp.add_argument("--no_label_free", action="store_true")
    sp.add_argument("--no_crosscheck", action="store_true")
    sp.set_defaults(func=cmd_medweb)

    sp = sub.add_parser("beir")
    common(sp)
    sp.add_argument("--dataset", required=True, help="e.g. fiqa, cqadupstack/android, chatdoctor, cmedqa")
    sp.add_argument("--data_dir", required=True)
    sp.add_argument("--split", default="test")
    sp.add_argument("--lang", default="en")
    sp.add_argument("--num_topics", type=int, default=50)
    sp.add_argument("--max_docs", type=int, default=0)
    sp.add_argument("--max_queries", type=int, default=0)
    sp.add_argument("--chunk_size", type=int, default=512)
    sp.add_argument("--download", action="store_true")
    sp.add_argument("--gamma_sweep", default="", type=lambda s: [float(x) for x in s.split(",") if x])
    sp.add_argument("--tau_sweep", default="", type=lambda s: [float(x) for x in s.split(",") if x])
    sp.set_defaults(func=cmd_beir)

    sp = sub.add_parser("export_cmedqa")
    sp.add_argument("--out_dir", required=True)
    sp.add_argument("--n_docs", type=int, default=10000)
    sp.set_defaults(func=cmd_export_cmedqa)

    sp = sub.add_parser("macros")
    sp.add_argument("--out", default="camera_ready_results")
    sp.set_defaults(func=cmd_macros)

    sp = sub.add_parser("selftest")
    sp.add_argument("--out", default="camera_ready_results")
    sp.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    if args.cmd not in ("macros", "selftest", "export_cmedqa") and _find_lextm_dir() is None:
        sys.exit("lex_tm_model.py not found: run from pipelines/v3_patched or set LEXTM_DIR")
    if args.cmd == "selftest":
        _find_lextm_dir()
    args.func(args)


if __name__ == "__main__":
    main()
