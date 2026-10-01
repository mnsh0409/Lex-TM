# TAU_VALIDATION_PROTOCOL.md — how τ was chosen, and a protocol for future work

## What the paper did

- **Review version.** MedWeb's 39 label sets were split into 32 test and 7
  validation sets. Summed over the three partitions, validation MRR favoured
  τ = 0.8 (2.396 against 2.347 for τ = 0.5; paper Table 6), although τ = 0.5
  scored higher on the Japanese and Chinese validation sets. That version's
  English headline nevertheless used τ = 0.5 (`FIXES.md` #27).
- **Camera-ready version.** τ = 0.8 for every partition, and all 39 label sets
  are reported, including the seven validation sets. τ was therefore not
  chosen strictly out of sample, and γ = 0.7 was fixed after sweeps on all
  label sets (`FIXES.md` #28). The paper reports both sweeps in full
  (Table 3): across τ ∈ {0.5, 0.8, 1.1}, MRR changes by at most 0.0014 in
  English and Chinese and 0.0124 in Japanese, less than it changes across
  Gibbs seeds 42–46. On MedWeb the choice of τ cannot drive the results.
- **This protocol was not run for the camera-ready version.** It is the plan
  for any collection where τ appears to matter.

## Why a protocol is still needed

While the CmedqaRetrieval benchmark was being added, an informal sweep of
τ ∈ {0.1, 0.4, 0.5, 0.8} on its 10,000-document subsample showed a
monotonic-looking trend (MRR 0.2344 → 0.2379 as τ decreased), while the same
sweep on MedWeb showed no such trend (HHI on/off made <0.1pp difference in
every language). The observation came from watching test-set results across
several runs, the practice `FIXES.md` #9 warns against, so it was not
reported. The steps below are how to test it properly.

**Do not skip Step 0.** Everything below is uninterpretable until it's fixed.

---

## Step 0 — fix Gibbs sampler determinism (blocking, do first)

`lex_tm_model.py` seeds only the first chunk of the flattened corpus
(`numba_seed = -1` for all subsequent chunks — `FIXES.md` #38). This means **rerunning the identical τ gives a
different MRR each time.** Without fixing this, "τ has an effect" and "the
sampler is noisy" cannot be told apart, which is the entire question this
protocol exists to answer.

Two acceptable fixes, pick one:
1. Derive a per-chunk seed deterministically from a single master seed
   (e.g. `chunk_seed = master_seed + chunk_index`) and pass it explicitly to
   every `fast_gibbs_loop` call, not just the first.
2. Set `checkpoint_size = max_iter` so the whole corpus is processed in one
   seeded call, with no chunk boundary to reseed at. Simpler, but loses the
   periodic convergence-checkpoint logging.

**Verification before proceeding:** run the identical config twice and
confirm bit-identical (or near-identical, <1e-6) MRR both times.

```bash
python main_pipeline1_patched.py --dataset_name cmedqa --gamma 0.7 --tau 0.8 \
    --num_topics 50 --filepath ... --run_bm25 > run_a.log
python main_pipeline1_patched.py --dataset_name cmedqa --gamma 0.7 --tau 0.8 \
    --num_topics 50 --filepath ... --run_bm25 > run_b.log
diff <(grep "^MRR:" run_a.log) <(grep "^MRR:" run_b.log)
# must be empty (or numerically negligible) before continuing
```

---

## Step 1 — fixed validation/test query split (make once, never regenerate)

The `cmedqa` export has no validation/test split; the informal sweep used
all 3,999 queries as "test". Create a genuine split, once,
with a fixed seed, and never touch it again for the life of this study.

```python
# split_cmedqa_val_test.py — run ONCE, commit the two output files to git
import csv, random

SEED = 20260713  # fixed — do not change once committed
VAL_FRACTION = 0.30

random.seed(SEED)
with open("../data/beir/cmedqa/qrels/test.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

query_ids = sorted({r["query-id"] for r in rows})  # sorted for determinism
random.shuffle(query_ids)
n_val = int(len(query_ids) * VAL_FRACTION)
val_ids = set(query_ids[:n_val])
test_ids = set(query_ids[n_val:])

with open("../data/beir/cmedqa/qrels/val.tsv", "w") as fval, \
     open("../data/beir/cmedqa/qrels/test_locked.tsv", "w") as ftest:
    fval.write("query-id\tcorpus-id\tscore\n")
    ftest.write("query-id\tcorpus-id\tscore\n")
    for r in rows:
        line = f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n"
        (fval if r["query-id"] in val_ids else ftest).write(line)

print(f"val queries: {len(val_ids)}  test queries: {len(test_ids)}")
print("Commit qrels/val.tsv and qrels/test_locked.tsv to git NOW, before any")
print("experiment touches them. test_locked.tsv is opened exactly once, at")
print("Step 4, and never before.")
```

---

## Step 2 — pre-register the grid and the selection rule BEFORE running anything

Fill in this block, commit it to git with a timestamp, before executing a
single experiment. This is what makes the eventual τ choice a real decision
rather than a post-hoc rationalization.

```
PRE-REGISTRATION — fill in and commit before Step 3
=====================================================
Date/time:            __________________
Tau grid:              {0.1, 0.3, 0.5, 0.7, 0.9, 1.1}
Repeats per tau:        5   (independent Gibbs seeds, see Step 3)
Selection rule:        ONE OF:
  [ ] highest mean validation MRR
  [ ] one-SE rule: simplest tau (closest to 1.1) within 1 SE of the best
Statistical test:      Jonckheere-Terpstra trend test (ordered alternative,
                        since tau is an ordered variable and the hypothesis
                        is directional: "more amplification changes MRR
                        monotonically", not just "the taus differ")
Significance threshold: alpha = 0.05
Test set touch policy: test_locked.tsv opened EXACTLY ONCE, after tau is
                        locked from validation results alone. No config
                        comparison on test, ever, for any reason.
```

---

## Step 3 — repeated runs per τ, validation split ONLY

For each τ in the pre-registered grid, run 5 independent Gibbs seeds on
`qrels/val.tsv`. This is the step the informal sweep skipped: without repeats,
four points that drift one direction cannot be distinguished from noise
(its four-point CIs overlapped heavily despite the visual trend).

```bash
for TAU in 0.1 0.3 0.5 0.7 0.9 1.1; do
  for SEED in 1 2 3 4 5; do
    python main_pipeline1_patched.py --dataset_name cmedqa \
        --filepath     ../data/beir/cmedqa/corpus.jsonl \
        --queries_path ../data/beir/cmedqa/queries.jsonl \
        --qrels_path   ../data/beir/cmedqa/qrels/val.tsv \
        --gamma 0.7 --tau $TAU --num_topics 50 --gibbs_seed $SEED \
        --query_smoothing 0.0 --no_adaptive_gamma \
        2>&1 | tee experiment_logs1/tauval_${TAU}_seed${SEED}.log
  done
done
```

**Note:** for MedWeb, `camera_ready_extras.py medweb --seeds 42,43,44,45,46`
already repeats the topic model with several Gibbs seeds. In
`main_pipeline1_patched.py`, `--gibbs_seed` does not currently exist as a CLI flag — it needs to
be added (threading a seed argument through to `LexTMLdaModel`'s
`random_state`, distinct from the `random_state=42` currently hardcoded at
line ~298 of `main_pipeline1_patched.py`) as part of Step 0's determinism
fix, since repeats are meaningless if every "repeat" produces the same
deterministic result.

### Analysis — is the trend real?

```python
import numpy as np
from scipy.stats import theilslopes
# Jonckheere-Terpstra is not in scipy directly; use the R-equivalent via
# rpy2, or the pure-Python implementation in scikit-posthocs
# (scikit_posthocs.posthoc_jonckheere or similar ordered-alternative test).
# Minimal fallback if unavailable: one-way ANOVA (ignores ordering, weaker)
# from scipy.stats import f_oneway
# f_oneway(*[mrr_by_tau[t] for t in sorted(mrr_by_tau)])

# Report: F-statistic / JT-statistic, p-value, and the validation MRR
# curve with error bars (mean +/- SE across the 5 seeds) per tau.
```

This curve — validation MRR vs. τ, with real error bars from repeated
seeds — is legitimate to report on its own, regardless of what happens
next, because nothing about it involved looking at test data.

---

## Step 4 — lock τ, touch test exactly once

Apply the pre-registered selection rule (Step 2) to the Step 3 validation
results. Whatever τ that rule outputs — even if it's not the one that
looked best in the informal sweep — is the final τ. Run it once on
`qrels/test_locked.tsv`:

```bash
python main_pipeline1_patched.py --dataset_name cmedqa \
    --filepath     ../data/beir/cmedqa/corpus.jsonl \
    --queries_path ../data/beir/cmedqa/queries.jsonl \
    --qrels_path   ../data/beir/cmedqa/qrels/test_locked.tsv \
    --gamma 0.7 --tau <LOCKED_VALUE> --num_topics 50 \
    --query_smoothing 0.0 --no_adaptive_gamma --run_bm25 \
    2>&1 | tee experiment_logs1/tau_final_test.log
```

Report this number. Do not compare it against what other τ values would
have scored on test — that comparison is exactly the practice this whole
protocol exists to prevent.

---

## Time budget

6 τ values × 5 repeats × ~24 min/run (at the 10K-doc cmedqa scale) ≈ 12
hours of compute for Step 3, plus the fixed cost of Steps 0/1/2 and one
final run in Step 4. Plan for a few days of calendar time, not a single
sitting.

---

## What not to do

- Do not compare configs on the test split before τ is locked.
- Do not add τ values to the grid mid-study because an early one "almost
  worked."
- Do not skip Step 0 and interpret single-seed runs as if they were stable.
- Do not report only the τ values that came out significant.
