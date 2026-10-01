# Technical Task: Gibbs Sampler Reproducibility Fix

**Status:** Open. Not applied in this release: `lex_tm_model.py` is unchanged.
The camera-ready paper reports every main result with Gibbs seed 42 and
measures the sampler's effect by repeating MedWeb with seeds 43–46
(Section 6.1); its paired tests are computed within one run, so they do not
depend on run-to-run variance. Bit-exact repetition of the camera-ready run
has not been verified.
**Priority:** High for exact reproducibility.
**Scope:** One file (`lex_tm_model.py`), one function. No change to the
model, the math, or any reported result — this makes existing runs
*reproducible*, it does not change what they compute in expectation.

---

## 1. Problem statement

The collapsed Gibbs sampler does not produce identical output across runs of
an identical configuration, even though a fixed `random_state` is passed.
Observed effect: rerunning the same `(dataset, τ, K, seed)` yields MRR values
that differ by roughly 0.01–0.03. This was the suspected cause of the review
version's Table 1 non-reproducibility documented in `FIXES.md` (fresh runs
landed 0.01–0.03 away from the manuscript's numbers, which was enough to move
Japanese and Chinese from "above RRF" to "at/below RRF").

### Why this matters

- **For a public release:** a reader who clones the repo and runs it may not
  reproduce the paper's Gibbs-dependent numbers exactly. This is the single
  largest gap between "verifiable" and "trivially reproducible by anyone".
- **For the method itself:** non-determinism also confounds any hyperparameter
  study (e.g. the τ dose-response question), because between-configuration
  differences cannot be cleanly separated from within-configuration run
  variance. See `TAU_VALIDATION_PROTOCOL.md`, which lists this fix as its own
  blocking Step 0.

---

## 2. Root cause

In `lex_tm_model.py`, the sampler runs in checkpointed chunks (to allow
periodic convergence checks in Python between bursts of Numba-compiled
iterations). The seeding logic is:

```python
# lines ~316–330 (current, buggy)
numba_seed = self.random_state if self.random_state is not None else -1
...
for start_iter in range(0, max_iter, checkpoint_size):
    ...
    topics, ... = fast_gibbs_loop(..., numba_seed)   # fast_gibbs_loop does
                                                     # np.random.seed(seed)
                                                     # iff seed >= 0
    numba_seed = -1   # <-- BUG: seed only the first chunk
```

`fast_gibbs_loop` (the `@njit` function) reseeds NumPy's RNG **only when
`random_seed >= 0`**. So:

- The **first** chunk is seeded deterministically (`numba_seed = random_state`).
- **Every subsequent** chunk is called with `numba_seed = -1`, i.e. it does
  NOT reseed and instead continues from whatever RNG state the Numba
  `nopython` context currently holds.

The intent was reasonable — reseeding every chunk to the *same* value would
reset the RNG and destroy mixing (every chunk would replay identical random
draws). But the chosen fix (`-1` = "don't reseed") is not deterministic:
**Numba's per-thread RNG state is not guaranteed to be carried across
separate `fast_gibbs_loop` invocations in a reproducible way.** The state at
the start of chunk *n+1* depends on JIT/threading details, not solely on the
master seed. Hence identical configs diverge after the first checkpoint.

---

## 3. Fix — recommended option

**Reseed every chunk, but with a DIFFERENT, deterministically-derived seed
per chunk.** This preserves mixing (each chunk draws a fresh, non-repeating
random stream) while making the entire run a pure function of the master
seed.

```python
# lines ~316–330 (proposed)
master_seed = self.random_state if self.random_state is not None else -1
...
for chunk_index, start_iter in enumerate(range(0, max_iter, checkpoint_size)):
    iters_this_chunk = min(checkpoint_size, max_iter - start_iter)
    current_iter     = start_iter + iters_this_chunk

    # Deterministic per-chunk seed: distinct for each chunk (preserves
    # mixing — chunks don't replay the same draws) yet fully determined by
    # master_seed (so the whole run is reproducible). When master_seed < 0
    # (caller opted out of seeding), pass -1 through unchanged.
    chunk_seed = (master_seed + chunk_index) if master_seed >= 0 else -1

    topics, doc_topic_counts, word_topic_counts, topic_counts = fast_gibbs_loop(
        docs, words, topics,
        doc_topic_counts, word_topic_counts, topic_counts,
        self.alpha, self.beta_vector, beta_sum,
        self.num_topics, iters_this_chunk,
        chunk_seed
    )
    pbar.update(iters_this_chunk)
```

The only functional change from the buggy version is the line computing
`chunk_seed` in place of the `numba_seed = -1` assignment. `fast_gibbs_loop`
itself is unchanged.

### Why not simply seed once and never reseed?

Because that is exactly what the current code *tries* to do (`-1` after the
first chunk = "don't reseed"), and it is precisely what fails — the
cross-chunk RNG state is not reproducible. Explicitly reseeding each chunk
from a derived value removes the dependence on carried state entirely.

### Alternative option (simpler, one trade-off)

Set `checkpoint_size = max_iter` so the whole sampler runs in a single seeded
`fast_gibbs_loop` call. Then there is only one seeding event and no
cross-chunk boundary at all — trivially deterministic.

- **Pro:** minimal change, obviously correct.
- **Con:** loses the periodic Python-level convergence checkpointing (the
  `min_iter` / `tol` early-stopping logic between chunks). For a fixed
  `max_iter=500` run this costs nothing but the intermediate logging; for
  runs relying on early stopping it removes that feature.

**Recommendation:** use the per-chunk-seed option (§3 main), which keeps
checkpointing intact. Fall back to the single-call option only if the
per-chunk fix somehow still shows variance in verification (§4), which would
indicate a second, deeper non-determinism source (e.g. Numba parallel
reduction order) worth investigating separately.

---

## 4. Verification (required before closing the task)

Determinism must be demonstrated, not assumed.

```bash
# Same config, twice. MRR must be identical (or <1e-9 apart).
for RUN in a b; do
  python main_pipeline1_patched.py --dataset_name medweb_en \
      --filepath <path>/medweb_rag_en_fixed.csv \
      --gamma 0.7 --tau 0.8 --num_topics 20 --chunk_size 512 \
      --query_smoothing 0.0 --adaptive_gamma \
      2>&1 | tee determinism_check_${RUN}.log
done
diff <(grep "^MRR:" determinism_check_a.log) \
     <(grep "^MRR:" determinism_check_b.log)
# REQUIRED: empty diff. Delete the cached topic distribution between the two
# runs (rm experiment_logs1/tdist_medweb_en_tau0.8_K20.npy) so the sampler
# actually re-executes rather than loading a cached result — otherwise the
# test is vacuous.
```

Additional checks:
1. **Different seeds still differ.** Run with two different `random_state`
   values; MRR should differ (confirms the seed is actually driving the RNG,
   not being ignored).
2. **Mixing is preserved.** Confirm final log-likelihood is comparable to
   pre-fix runs (the fix should not degrade convergence — if LL is markedly
   worse, the per-chunk reseeding is disrupting mixing and needs revisiting).

---

## 5. Downstream once the fix lands

- **Regenerate the Gibbs-dependent MedWeb numbers** from deterministic runs and
  check them against `results/camera_ready/`; record any difference in
  `FIXES.md`.
- **State in future versions** that all results are reproducible with a
  given seed, once verified.
- **Unblocks `TAU_VALIDATION_PROTOCOL.md` Step 0**, enabling the proper
  repeated-seed τ study for future work.

---

## 6. Explicit scope boundary

This fix changes reproducibility, not results. In expectation the sampler
computes the same thing before and after; the fix only removes run-to-run
variance around that expectation. It can therefore be applied without
re-opening any scientific claim, but the Gibbs-dependent numbers will move
within the seed-to-seed range the paper reports (Section 6.1), so they must be
regenerated, not assumed.
