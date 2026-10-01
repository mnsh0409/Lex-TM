# Outputs of the camera-ready run

These files are the outputs behind every camera-ready number that
`docs/REPRODUCING_TABLES.md` attributes to the camera-ready run. The run
finished on 27 Sep 2026 on one RTX 4090 (Windows) with the four blocks of
`scripts/run_camera_ready.sh`.

| File | Content |
|---|---|
| `medweb_results.json` | MedWeb EN/JA/ZH: every system with MRR@20, 95% CI and Recall@5/10/20; paired tests; γ and τ sweeps; Gibbs seeds 42–46; query-type breakdown; Figure 1 candidates; label-free control; length statistics; and **per-query reciprocal ranks** for every system (`langs.<lang>.per_query`, in the order of `langs.<lang>.queries`) |
| `per_query/medweb_{en,ja,zh}.csv` | the same per-query reciprocal ranks, one row per label set: `query`, `n_relevant`, then one column per system |
| `beir_fiqa_results.json`, `beir_chatdoctor_results.json`, `beir_cmedqa_results.json` | FiQA-2018, ChatDoctor and CmedqaRetrieval: every system with nDCG@10, query-level MRR@10 and pair-level MRR@20; paired tests against Lex-TM with win/loss/tie counts, intervals and p-values; collection statistics |
| `results_generated.tex` | the macro file the paper's LaTeX source reads; rebuilt from the JSON files by `camera_ready_extras.py macros` |
| `logs/` | console logs of the four blocks |

`config` in each JSON file records the command-line configuration of the run.
Three edits were made for release:

- absolute paths of the machine that ran it are replaced by `<DATA>` (data
  root), `<RESULTS>` (output directory), `<home>`, `<tmp>` and
  `<site-packages>` in the JSON files and logs;
- MedWeb is distributed under the NTCIR data agreement, so the document texts
  in `figure1_candidates` are removed (chunk ids identify the documents),
  except for the English case printed in Figure 1
  (`langs.en.figure1_candidates[1]`);
- `results_generated.tex` was rebuilt from these JSON files with the released
  `FIG1_EN=1 camera_ready_extras.py macros`. It equals the macro file the
  paper was compiled with, except for the timestamp and the Japanese and
  Chinese Figure 1 candidate texts, which the paper does not use. (The paper's
  macro file was itself rebuilt from the run's JSON after the run, which is why
  it holds more macros than the run's last log line reports.)

`per_query/*.csv` is written by `scripts/export_per_query_csv.py`.

## Code version

The run used `pipelines/v3_patched/` as it stood on 11 July 2026 together with
`camera_ready_extras.py`. The released `v3_patched` files include the changes
made later in July (#19b, #21, #22) and the comment edits of this release;
none touches the scoring path:
`lex_tm_model.py` differs only in four comment and error-message lines (the
`tau_mode='adaptive'` stub); `rag_evaluation_pipeline_patched.py` adds
the paired bootstrap helper and an opt-in `trust_remote_code` switch;
`main_pipeline1_patched.py` adds the CmedqaRetrieval loader, prints the paired
tests after `--run_bm25`, and exposes `--trust_remote_code`; `rag_dataload.py`
prints the BEIR dataset name instead of a fixed "NFCorpus" label
(`docs/FIXES.md` #22). The released
`camera_ready_extras.py` additionally writes the per-query files described
below; its scoring code is unchanged.

## What this run did not save

The run kept per-query values for the MedWeb systems only. For FiQA-2018,
ChatDoctor, CmedqaRetrieval and the MedWeb label-free control it saved
aggregate metrics and paired statistics, but not the per-query values behind
them (`docs/FIXES.md` #36). The released `camera_ready_extras.py` writes them
to `per_query/beir_<name>.json` and `per_query/medweb_<lang>_label_free.json`;
rerunning blocks 2–4 of `scripts/run_camera_ready.sh` (or block 1 for the
label-free control) produces them.
