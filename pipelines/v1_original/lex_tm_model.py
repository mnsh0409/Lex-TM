import numpy as np
from scipy.sparse import csr_matrix, diags
from numba import jit
from tqdm import tqdm
 
# ==========================================
# PURE MATH NUMBA KERNELS (NO PYTHON CLASSES)
# ==========================================
 
@jit(nopython=True)
def fast_gibbs_loop(docs, words, topics, doc_topic_counts, word_topic_counts,
                    topic_counts, alpha, beta_vector, beta_sum, num_topics,
                    num_iters, random_seed):
    """
    C-compiled Gibbs Sampler using flat 1D arrays.
 
    SPEED-UP 1 (Cache locality): word_topic_counts has shape (V, K) rather
    than (K, V). The hot-path access for word w is word_topic_counts[w, :]
    — a contiguous row read — instead of the old topic_word_counts[:, w]
    column read which caused a cache miss every K steps. For V=50,000 that
    was 200KB of memory jumps per token; now it is a single 200-byte read.
    Estimated speedup: 2–4× on large vocabularies.
 
    The array is renamed word_topic_counts (V, K) throughout to make the
    transposition explicit and avoid confusion with the old (K, V) shape.
 
    FIX 1 (Reproducibility): random_seed seeded inside nopython JIT context.
    FIX 2 (Performance): all print() calls removed from the hot loop.
    """
    if random_seed >= 0:
        np.random.seed(random_seed)
 
    num_tokens = len(topics)
 
    for it in range(num_iters):
        for i in range(num_tokens):
            d = docs[i]
            w = words[i]
            z = topics[i]
 
            # 1. Decrement counts
            doc_topic_counts[d, z] -= 1
            word_topic_counts[w, z] -= 1   # (V,K): row w, column z
            topic_counts[z] -= 1
 
            # 2. Calculate unnormalised probabilities
            # word_topic_counts[w, :] is a contiguous row read — cache friendly
            p_topic = np.zeros(num_topics)
            for k in range(num_topics):
                p_topic[k] = (
                    (word_topic_counts[w, k] + beta_vector[w]) /
                    (topic_counts[k] + beta_sum)
                ) * (doc_topic_counts[d, k] + alpha)
 
            # 3. Normalise
            p_topic = p_topic / np.sum(p_topic)
 
            # 4. Sample via cumulative sum
            r = np.random.rand()
            cum_p = 0.0
            new_z = num_topics - 1
            for k in range(num_topics):
                cum_p += p_topic[k]
                if r < cum_p:
                    new_z = k
                    break
 
            # 5. Increment counts
            topics[i] = new_z
            doc_topic_counts[d, new_z] += 1
            word_topic_counts[w, new_z] += 1   # (V,K): row w, column new_z
            topic_counts[new_z] += 1
 
    return topics, doc_topic_counts, word_topic_counts, topic_counts
 
 
@jit(nopython=True)
def compute_log_likelihood(word_topic_counts, doc_topic_counts, topic_counts,
                           beta_vector, beta_sum, alpha, num_topics):
    """
    Computes the approximate joint log-likelihood of the current Gibbs state.
    Called from Python between checkpoints to detect convergence.
 
    SPEED-UP 1 (Cache locality): loop order swapped to v-outer, k-inner so
    that word_topic_counts[v, :] is read as a contiguous row for each v,
    matching the (V, K) memory layout.
 
    LL ≈ Σ_v Σ_k n_vk * log P(v|k)  +  Σ_d Σ_k n_dk * log P(k|d)
    """
    ll = 0.0
    vocab_size = word_topic_counts.shape[0]   # V (first dim of V×K array)
    num_docs   = doc_topic_counts.shape[0]
 
    # Token-topic term — v outer, k inner → row access word_topic_counts[v, :]
    for v in range(vocab_size):
        for k in range(num_topics):
            count = word_topic_counts[v, k]
            if count > 0:
                denom = topic_counts[k] + beta_sum
                ll += count * np.log((count + beta_vector[v]) / denom)
 
    # Document-topic term — unchanged
    for d in range(num_docs):
        doc_total = np.sum(doc_topic_counts[d]) + num_topics * alpha
        for k in range(num_topics):
            count = doc_topic_counts[d, k]
            if count > 0:
                ll += count * np.log((count + alpha) / doc_total)
 
    return ll
 
 
# ==========================================
# MAIN LEX-TM CLASS
# ==========================================
 
class LexTMLdaModel:
    """
    Lexically-Guided Dirichlet Prior (Lex-TM) Topic Model.
    """
    def __init__(self, num_topics, alpha=0.1, beta_base=0.01,
                 tau=0.5, lambda_amp=2.0, random_state=None):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta_base = beta_base
        self.tau = tau
        self.lambda_amp = lambda_amp
        self.random_state = random_state   # stored; passed into Numba at fit() time
 
        self.dictionary = None
        self.vocab_size = 0
        self.beta_vector = None
        self.exclusivity_scores = None
        self.topic_word_dist = None
        self.doc_topic_dist = None
 
    def _compute_dynamic_prior(self, corpus, doc_assignments=None,
                               hhi_scope='document', tau_mode='fixed'):
        """
        Computes the asymmetric beta vector by scaling each token's prior
        mass by its Herfindahl-Hirschman exclusivity score.
 
        SPEED-UP 2 (Sparse CSR matrix): The original used nested Python
        defaultdict loops — O(V × N_docs) pure-Python operations. This
        version builds a scipy sparse (V × N_scopes) count matrix in a
        single pass, then computes HHI as vectorised row operations:
            - Row normalisation via sparse diagonal scaling
            - HHI = row-wise sum of squares via .power(2).sum(axis=1)
        Estimated speedup: 10–100× on large corpora.
 
        hhi_scope : 'document' — aggregate counts at document level (default)
                    'chunk'    — treat each chunk as its own scope (ablation)
 
        tau_mode  : 'fixed'    — use self.tau as a scalar threshold
                    'adaptive' — reserved; not implemented
        """
        print("Calculating Lexical Exclusivity (HHI) via sparse CSR matrix...")
 
        if doc_assignments is None:
            print(
                "WARNING: doc_assignments not provided. HHI will be computed "
                "at CHUNK level. Pass doc_assignments for paper-accurate scores."
            )
 
        # ------------------------------------------------------------------
        # Build COO data for sparse (V × N_scopes) token-document count matrix
        # ------------------------------------------------------------------
        rows_idx, cols_idx, data_vals = [], [], []
 
        for chunk_idx, doc in enumerate(corpus):
            scope_id = (
                doc_assignments[chunk_idx]
                if (doc_assignments is not None and hhi_scope == 'document')
                else chunk_idx
            )
            for word_id, count in doc:
                rows_idx.append(word_id)
                cols_idx.append(scope_id)
                data_vals.append(float(count))
 
        n_scopes = (
            int(max(doc_assignments)) + 1
            if (doc_assignments is not None and hhi_scope == 'document')
            else len(corpus)
        )
 
        # Build sparse matrix — duplicate (word_id, scope_id) entries are
        # summed automatically by csr_matrix construction
        X = csr_matrix(
            (data_vals, (rows_idx, cols_idx)),
            shape=(self.vocab_size, n_scopes),
            dtype=np.float64,
        )
 
        # ------------------------------------------------------------------
        # Vectorised HHI: HHI_v = Σ_d P(d|v)²  where P(d|v) = n(v,d)/n(v)
        # ------------------------------------------------------------------
        # Total count per word (row sums) — shape (V,)
        total_word_counts = np.array(X.sum(axis=1)).flatten()
 
        # Safe reciprocal: words that never appear get HHI = 0
        inv_totals = np.zeros(self.vocab_size)
        nonzero    = total_word_counts > 0
        inv_totals[nonzero] = 1.0 / total_word_counts[nonzero]
 
        # Normalise rows: X_norm[v, d] = P(d|v)
        # diags(inv_totals) @ X scales each row v by inv_totals[v]
        X_norm = diags(inv_totals) @ X
 
        # HHI = row-wise sum of squares — shape (V,)
        hhi_scores = np.array(X_norm.power(2).sum(axis=1)).flatten()
 
        # ------------------------------------------------------------------
        # Normalise HHI to [0, 1] and compute dynamic beta vector
        # ------------------------------------------------------------------
        hhi_min, hhi_max = hhi_scores.min(), hhi_scores.max()
        if hhi_max > hhi_min:
            e_v = (hhi_scores - hhi_min) / (hhi_max - hhi_min)
        else:
            e_v = np.zeros(self.vocab_size)
 
        self.exclusivity_scores = e_v
 
        # tau_mode='adaptive' is a stub and is not implemented.
        if tau_mode == 'fixed':
            mask = e_v > self.tau
        else:
            raise NotImplementedError(
                "tau_mode='adaptive' is reserved and not implemented."
            )
 
        beta_vector = np.full(self.vocab_size, self.beta_base)
        beta_vector[mask] = self.beta_base * (1.0 + self.lambda_amp * e_v[mask])
 
        print(
            f"Prior scaling complete. "
            f"{int(np.sum(mask))}/{self.vocab_size} specialised tokens amplified."
        )
        return beta_vector
 
    def fit(self, corpus, dictionary, max_iter=500,
            convergence_threshold=1e-4, min_iter=50,
            checkpoint_size=50, doc_assignments=None,
            hhi_scope='document', tau_mode='fixed'):
        """
        Trains the Lex-TM model via Collapsed Gibbs Sampling.
 
        Parameters
        ----------
        corpus               : list of list of (int, int)  chunk-level BoW
        dictionary           : dict {int: str}  id2token from the loader
        max_iter             : int   hard upper bound on iterations
        convergence_threshold: float stop early if |ΔLL| < threshold
        min_iter             : int   burn-in — no convergence check before this
        checkpoint_size      : int   iterations per Python-level checkpoint
        doc_assignments      : list of int  chunk→document ID for document HHI
        hhi_scope            : 'document' | 'chunk'  HHI aggregation level
        tau_mode             : 'fixed' | 'adaptive'  threshold mode (adaptive
                               reserved; not implemented)
        """
        self.dictionary  = dictionary
        self.vocab_size  = len(dictionary)
 
        if self.vocab_size == 0:
            raise ValueError(
                "Vocabulary size is 0. Check data loader thresholds."
            )
 
        num_docs = len(corpus)
 
        # 1. Dynamic prior via sparse CSR HHI
        self.beta_vector = self._compute_dynamic_prior(
            corpus, doc_assignments, hhi_scope, tau_mode
        )
        beta_sum = np.sum(self.beta_vector)
 
        # 2. Flatten corpus into 1D arrays for Numba
        print("Flattening corpus for Numba C-compilation...")
        doc_list, word_list, topic_list = [], [], []
 
        doc_topic_counts  = np.zeros((num_docs,        self.num_topics), dtype=np.int32)
        # SPEED-UP 1: (V, K) layout — row access word_topic_counts[w, :]
        # is cache-friendly in the Gibbs hot loop
        word_topic_counts = np.zeros((self.vocab_size, self.num_topics), dtype=np.int32)
        topic_counts      = np.zeros(self.num_topics, dtype=np.int32)
 
        if self.random_state is not None:
            np.random.seed(self.random_state)
 
        for d, doc in enumerate(corpus):
            for word_id, count in doc:
                for _ in range(count):
                    topic = np.random.randint(self.num_topics)
                    doc_list.append(d)
                    word_list.append(word_id)
                    topic_list.append(topic)
 
                    doc_topic_counts[d, topic]          += 1
                    word_topic_counts[word_id, topic]   += 1   # (V,K)
                    topic_counts[topic]                 += 1
 
        docs   = np.array(doc_list,   dtype=np.int32)
        words  = np.array(word_list,  dtype=np.int32)
        topics = np.array(topic_list, dtype=np.int32)
 
        # 3. Chunked Gibbs with convergence checking
        print(
            f"Starting Numba Gibbs Sampling "
            f"(max_iter={max_iter}, min_iter={min_iter}, "
            f"checkpoint={checkpoint_size}, tol={convergence_threshold})..."
        )
 
        prev_ll    = -np.inf
        converged  = False
        numba_seed = self.random_state if self.random_state is not None else -1
 
        with tqdm(total=max_iter, desc="Gibbs Sampling", unit="iter") as pbar:
            for start_iter in range(0, max_iter, checkpoint_size):
                iters_this_chunk = min(checkpoint_size, max_iter - start_iter)
                current_iter     = start_iter + iters_this_chunk
 
                topics, doc_topic_counts, word_topic_counts, topic_counts = fast_gibbs_loop(
                    docs, words, topics,
                    doc_topic_counts, word_topic_counts, topic_counts,
                    self.alpha, self.beta_vector, beta_sum,
                    self.num_topics, iters_this_chunk,
                    numba_seed
                )
                numba_seed = -1   # seed only the first chunk
                
                pbar.update(iters_this_chunk) # Advance progress bar
 
                if current_iter < min_iter:
                    pbar.set_postfix(Status="Burn-in")
                    #print(f"  Iteration {current_iter}/{max_iter} — burn-in phase.")
                    continue
 
                ll    = compute_log_likelihood(
                    word_topic_counts, doc_topic_counts, topic_counts,
                    self.beta_vector, beta_sum, self.alpha, self.num_topics
                )
                delta = abs(ll - prev_ll)
                #print(
                #    f"  Iteration {current_iter}/{max_iter} — "
                #    f"LL = {ll:.2f}, ΔLL = {delta:.6f}"
                #)
                
                # Dynamically update the right side of the progress bar
                pbar.set_postfix(LL=f"{ll:.2f}", dLL=f"{delta:.6f}")
 
                if delta < convergence_threshold:
                    print(f"Converged at iteration {current_iter} (ΔLL < {convergence_threshold}).")
                    converged = True
                    break
 
                prev_ll = ll
 
        if not converged:
            print(
                f"Reached max_iter={max_iter} without convergence. "
                "Consider increasing max_iter or relaxing convergence_threshold."
            )
 
        # 4. Final parameter estimation
        # word_topic_counts is (V, K); topic_word_dist must be (K, V) for
        # downstream use in get_document_topic_distribution and get_top_topic_words.
        # Transpose and apply smoothing in one vectorised step.
        #
        # word_topic_counts.T           shape (K, V)
        # + self.beta_vector            shape (V,)  → broadcasts to (K, V)
        # / (topic_counts[:,None]       shape (K, 1) + beta_sum)
        self.topic_word_dist = (
            (word_topic_counts.T + self.beta_vector) /
            (topic_counts[:, np.newaxis] + beta_sum)
        )
        self.doc_topic_dist = (
            (doc_topic_counts + self.alpha) /
            (doc_topic_counts.sum(axis=1)[:, np.newaxis] + self.num_topics * self.alpha)
        )
        print("Gibbs Sampling complete.")
 
    def get_document_topic_distribution(self, doc_bow):
        """
        Fast approximate topic inference for unseen documents and queries.
 
        Note: This is a weighted-sum approximation over the trained topic-word
        distributions, not full Gibbs inference. It is intentionally lightweight
        for real-time query routing. For very short queries (< 5 tokens), the
        resulting theta_q will be noisy — this is an inherent limitation of
        bag-of-words topic models on short text, not a bug.
        """
        doc_topic_counts = np.zeros(self.num_topics)
        for word_id, count in doc_bow:
            doc_topic_counts += self.topic_word_dist[:, word_id] * count
        return (doc_topic_counts + self.alpha) / (
            np.sum(doc_topic_counts) + self.num_topics * self.alpha
        )
 
    def get_top_topic_words(self, top_n=10):
        """
        Returns the top-N highest probability words for each topic.
        """
        top_topics = []
        for k in range(self.num_topics):
            top_word_indices = np.argsort(self.topic_word_dist[k])[-top_n:][::-1]
            top_words = [self.dictionary[idx] for idx in top_word_indices]
            top_topics.append(top_words)
        return top_topics
