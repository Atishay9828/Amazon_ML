# AJ's Kaggle dual T4 experiment

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

**Recovery verification:** Python syntax and notebook structure are checked locally. Recovery on AJ's actual Kaggle files remains unverified until that cell completes. If working files were lost, the recovery cell cannot regenerate them.

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
| Final TSV assembly | Kernel died before output completion; recovery pending |
| Portal / leaderboard | No submission status or leaderboard score supplied |

The 84.01% candidate-link recall means about 15.99% of labeled validation links never reached the matcher. This quality issue is separate from the export failure. Recovery preserves the existing predictions; it does not improve that recall or change the threshold. France is present in test but absent from labeled training, so its accuracy cannot be measured from the supplied labels. Earlier empty-Jaccard guards, use of the saved best iteration during inference, and the removal of package downgrades are retained.
