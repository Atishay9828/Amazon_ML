# AJ's Kaggle dual T4 experiment

## Preprocessing and tests

The new [preprocessing diagnostic notebook](aj_preprocessing_diagnostics.ipynb) has separate cells for setup, tests, retrieval, unlabeled test profiling, reporting, statistical uncertainty, and aggregate export; [preprocessing_diagnostics_cell.py](preprocessing_diagnostics_cell.py) is the standalone equivalent. Attaching AJ's existing `student_stuff` dataset on Kaggle reuses the stored dataset and does not upload it again. The notebook runs synthetic preprocessing tests, profiles the data, and compares retrieval variants against the full target-country populations. It does not fit models or create submission predictions. The final cells export aggregate-only JSON/CSV/figures and provide a direct browser download link; no raw records, sampled IDs, examples, candidate rows, or predictions are in that bundle. The [team findings note](TEAM_FINDINGS_RECONCILIATION.md) incorporates AJ's screenshots and distinguishes reported counts from guarantees that the evidence does not support.

The [preprocessing module](preprocessing.py) retains original fields and adds Unicode-safe name/address views, repeated trailing legal-suffix removal, conservative address abbreviation views, all address number runs, and explicit empty flags. Its extra views do not assign house-number meaning or fill missing business data.

The [diagnostic runner](diagnostics.py) selects 1,000 original development Source 1 records per training country, excluding the original validation buckets, and then freezes that membership for all comparisons. It measures where each labeled link is lost: no shared eligible key, full-target frequency filter, query-key budget, or final rank cap. Missing target addresses receive their own recall slice. Caps 40/80/160/500 are per source. The controlled ablation compares new-preprocessing retrieval at DF<=64 / 3 token keys per channel, DF-only at 256 / 3 keys, key-budget-only at 64 / 6 keys, and the combined 256 / 6 setting, isolating the effects of frequency tolerance and query-key budget. The report also combines Source 2 and Source 3 per Source 1 entity and reports micro/mean/median positive-query recall, fully/partially/uncovered positive queries, singleton rate, and the singleton-aware oracle macro F0.5 ceiling. That ceiling assumes a perfect classifier that drops every false candidate; it is a candidate-retrieval upper bound, not a model or leaderboard score. Candidate true/false shares describe label composition in this diagnostic sample and are not fitted-model precision.

Use the same official data and DuckDB 1.3.2 environment as the completed Kaggle run. The original split uses DuckDB's `hash()`, whose output can change across versions ([official documentation](https://duckdb.org/docs/current/sql/functions/utility#hashvalue)). No package installer or downgrade runs in the notebook. This controlled ablation writes to `aj_preprocessing_ablations/`, leaving the earlier `aj_preprocessing_diagnostics/` results intact. Private example rows and sample membership remain out of Git.

Run the seven code cells from top to bottom for the first audit. The default output path is now `work/aj_preprocessing_ablations/`, keeping the controlled-ablation artifacts separate from the earlier `work/aj_preprocessing_diagnostics/` report. The reporting cell reads the saved report and can redraw figures without generating candidates again. After an interruption, rerun setup, settings/tests, and retrieval while keeping the same working directory; the retrieval runner validates completed partition checkpoints before reusing them. The test-profile scan has its own cell. Input attachment and output preservation are separate: the dataset stays attached, but working files must be saved/downloaded before a Kaggle session is discarded.

Tests are in [test_preprocessing.py](tests/test_preprocessing.py), [test_diagnostics.py](tests/test_diagnostics.py), and [test_entity_metrics.py](tests/test_entity_metrics.py). The synthetic retrieval fixtures verify full-index frequencies, batched accumulation, stage attribution, cap failures even with one true match, missing-address failure cases, label-independent ranking, validation exclusion, and singleton-aware aggregation across both target sources. Run the suite under the isolated DuckDB 1.3.2 runtime for all tests; the normalization suite also supports the installed newer DuckDB. Regenerate the notebook using `python experiments/aj_kaggle/sync_diagnostics_notebook.py` after changing its source modules or preprocessing tests.

## Recover the September 26 export failure

Both GPU models and all six inference partitions completed in AJ's supplied log. The old next step was a global `string_agg(tid, ',' ORDER BY tid)` across the candidate files. DuckDB documents that `string_agg` aggregate states cannot spill to disk, and that some allocations bypass its memory limit. This makes host RAM exhaustion during export the leading explanation for the kernel death; no host kill log or memory trace was supplied to prove it. See the [DuckDB workload guide](https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads) and [OOM guidance](https://duckdb.org/docs/current/guides/performance/oom).

**Use [recover_outputs.py](recover_outputs.py) or the single code cell in [aj_recover_outputs.ipynb](aj_recover_outputs.ipynb) in the existing Kaggle session.**

1. Preserve `/kaggle/working/aj_entity_resolution/`. It contains the expensive completed work. A Python kernel restart may be needed; avoid deleting/resetting the session or working files.
2. Keep the official dataset attached. Recovery needs `dataset/test/test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv`, along with six Parquets in each of `test_candidates/` and `accepted/`.
3. Paste the **entire** contents of `recover_outputs.py` into a new code cell in that same notebook. Run **only that cell**. Do not rerun the original full training cell. Importing the recovery notebook into a separate Kaggle session does not transfer the original working directory; restore those saved files first if changing sessions.
4. Wait for `OUTPUT COMPLETE`. The cell prints row progress, process RAM, and free disk. It will stop on missing files, count mismatches, unknown IDs, duplicate pairs, or mismatched candidate/match sets.
5. Download `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and `output_completion.json` from the run directory. Only the first TSV is a portal upload. Preserve models, validation files, and the rest of the run for the final package.

The recovery cell reuses saved IDs and does no training or inference. It checks old accepted-file row counts against the six completed inference messages supplied by AJ: **4,953,551 accepted links** in total. Old files have no completion sidecars, so matching these counts is evidence from the supplied log; model and threshold provenance cannot be independently reconstructed from their filenames.

Assembly uses sequential external sorts of only `qid` and `tid`, with one DuckDB thread and a 768 MB buffer-manager budget. It merges small batches and writes one Source 1 entity at a time, including empty rows. Additional disk space is required for sorted ID caches, temporary spill, and final TSVs. The DuckDB budget is not a hard cap on total Python process RAM. A later recovery attempt can reuse completed sorted caches.

Both outputs receive checks for exact Source 1 coverage, target existence in the official test files, unique pairs, and final matches being a subset of candidates. They are first written to temporary paths; `output_completion.json` is written last and records hashes. The separate organizer validator is not run automatically because it builds large Python collections. This code's validation status is not an organizer `PASS` or leaderboard `SCORED` result.

**Recovery verification:** AJ supplied a Kaggle log reaching `OUTPUT COMPLETE` at `2026-09-26 04:41:11`. It reports all 1,732,544 Source 1 rows, 88,591,494 candidate pairs, and 4,953,551 accepted links, with the assembler's coverage, existence, uniqueness, and subset checks passing. Process RAM at completion was 0.24 GiB and disk free was 16.01 GiB; these are point-in-time readings, not peak measurements. This confirms the reported recovery run. The output files themselves have not been downloaded or independently checked here, the organizer validator did not run, and the revised full notebook still needs a fresh complete run to verify its new inference checkpoints.

**Current-session check, September 26:** the opened private `Amazon_ML` notebook still displays those saved recovery logs. A newly executed preflight in its live editor confirmed DuckDB 1.3.2, two Tesla T4 GPUs, and the attached official dataset, but found **no** `aj_entity_resolution/output_completion.json`, `output/matching_results.tsv`, or `output/candidate_pairs.tsv` in the current working directory. The historical log establishes the earlier run's reported completion, not that its files survive in this session. Preserve any copies AJ previously downloaded; the recovery cell requires the actual saved Parquets.

## Full run

Open `aj_dual_t4_entity_resolution.ipynb` on Kaggle and attach the official challenge download as a **private** dataset. Select the **GPU T4 x2** accelerator, then run its single code cell. It stops immediately if two CUDA devices are unavailable. The notebook finds the `student_resource/dataset/` folder automatically if it is the only attached copy.

The run creates `/kaggle/working/aj_entity_resolution/output/matching_results.tsv` and `candidate_pairs.tsv`, two XGBoost model files, validation score files, and a run manifest. Upload only `matching_results.tsv` to the live portal. The final team package also needs `candidate_pairs.tsv`, the integrated runnable source, pinned environment, and completed methodology document.

The experiment processes one country and target source at a time with DuckDB, keeps bounded name and address posting lists, and measures true-link candidate recall on a deterministic held-out sample. It trains a Source 2 matcher on `cuda:0` and a Source 3 matcher on `cuda:1` concurrently. Threshold selection uses the challenge's macro F0.5 formula over **every** held-out Source 1 entity, including those with no candidates or no true links. It retains any number of accepted matches per Source 1. The actual final candidate set is the exact set scored at inference.

The updated full notebook also uses the streaming assembler. It writes `validation_summary.json` before inference and uses temporary accepted Parquets followed by completion records, so interrupted inference is not mistaken for a complete checkpoint. Those records include candidate file metadata, scored/accepted counts, threshold, prediction rounds, and candidate/model/accepted-file hashes. Content hashes allow intact completed files to be recovered after moving the saved run directory.

The official data audit is in [DATA_AUDIT.md](DATA_AUDIT.md). Readable source lives in [one_cell_solution.py](one_cell_solution.py) and [output_assembly.py](output_assembly.py). The generator embeds both in exactly one code cell; no project files need to be imported on Kaggle. Edit those sources, then run `python experiments/aj_kaggle/sync_notebook.py` from the repository root to regenerate both notebooks and the standalone recovery cell. Keep challenge data, generated predictions, model files, and ZIPs out of this public repository.

## Measured status from AJ's September 26 log

| Item | Reported result |
| --- | --- |
| GPU training | Source 2 on GPU 0 and Source 3 on GPU 1 completed |
| Best iteration | 747 for both models |
| Selected threshold | Approximately 0.58 |
| Held-out macro F0.5 | 0.8336141058305573 |
| True-link candidate recall | 0.840123886140238 |
| Correctly empty singleton predictions | 5,228 / 6,343 |
| Test inference | All six partitions reported completion; 4,953,551 accepted pairs |
| Final TSV assembly | Recovery reported `OUTPUT COMPLETE` at 04:41:11 on September 26 |
| Source 1 rows in each output | 1,732,544 |
| Candidate pairs | 88,591,494 |
| Source 1 entities with no candidates | 13,791 |
| Assembly checks | PASS: coverage, target existence, unique pairs, match subsets |
| Organizer validator | Not run |
| Portal / leaderboard | No submission status or leaderboard score supplied |

Output identity from the recovery log (hashes have not been recomputed locally):

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `matching_results.tsv` | 86,350,245 | `6f4888501f02f8c6ea6e9dc304167614de5c5d8fbf2328bcccc3d00dc458c4b2` |
| `candidate_pairs.tsv` | 1,164,169,169 | `54c659e077f9e3863d595691be9d22e9e4fe43398aa1ad154c2d92464b7f62dc` |

The 84.01% candidate-link recall means about 15.99% of labeled validation links never reached the matcher. This quality issue is separate from the export failure. Recovery preserves the existing predictions; it does not improve that recall or change the threshold. France is present in test but absent from labeled training, so its accuracy cannot be measured from the supplied labels. Earlier empty-Jaccard guards, use of the saved best iteration during inference, and the removal of package downgrades are retained.
