# Amazon ML Challenge — end-to-end plan

**Plan date:** 26 September 2026
**Purpose:** Agree on the full route from the current experiments to a reproducible, quality-checked submission. This is a plan and evidence summary, not a claim that the remaining runs or portal submission are complete.

## Recommendation

Keep the problem as two linked stages: retrieve a bounded candidate set from the full target sources, then score every retained pair with a separate matcher. Improve and select retrieval first because the reported AJ baseline let only 84.01% of validation true links reach the matcher; a classifier cannot recover a true pair that retrieval dropped. Preserve the baseline as a fallback and comparison point.

Use a competition-style workflow: small, fixed query samples for fast controlled experiments against the complete target pools; multiple complementary retrieval channels; a pairwise tree model; validation on complete Source 1 entities; chunked, resumable full runs; then a clean inference and output-validation pass. Keep XGBoost model training on the requested two T4 GPUs. Candidate search itself may stay CPU/DuckDB if it wins on measured quality, memory, and elapsed time.

## Problem contract we will keep

- For every test Source 1 business, identify zero or more matches in Source 2 and Source 3.
- The official files are **matching_results.tsv** and **candidate_pairs.tsv**. Each contains every test Source 1 ID exactly once, including empty lists. Every final match must appear in the candidate list; IDs must exist and may not be duplicated within a list.
- Optimize the challenge's per-Source-1 macro F0.5, including entities with no matches. This is precision-heavy, so retrieval recall matters, but the final decision threshold must be selected on the real metric, not candidate recall alone.
- Preserve multiple matches. The supplied rules do not impose a global one-to-one assignment.
- Use only provided challenge data. Do not add outside business lookup, geocoding, test labels, or train/test association tricks.
- Keep the country set dynamic. Train has US and India; test also has 259,452 France Source 1 rows, and there is no labeled France training slice.
- Keep challenge records, row-level predictions, models, and private outputs out of the public Git repository.

The official format validator is a structural gate, not a measure of matching quality. We must report format validation and held-out quality as different results.

## What already works, and what is still evidence-limited

| Area | Current evidence | How we use it |
|---|---|---|
| Data and output contract | Local audit records 2,206,821 / 5,034,616 / 5,285,603 training rows and 1,732,544 / 4,887,273 / 5,082,316 test rows for Sources 1 / 2 / 3. It records 7,638,365 links, 123,247 singletons, and the France test slice. | Keep the audit as the shared source for schemas, counts, and output requirements. |
| AJ GPU baseline | A supplied Kaggle log reports two XGBoost models completed on T4 GPU 0 and GPU 1, held-out macro F0.5 of 0.833614, candidate-link recall of 0.840124, and 5,228 / 6,343 correctly empty singleton predictions. Its native cap is up to 40 candidates per target source (80 combined). | Retain it as the first model baseline. Re-run it on the new, disjoint model-validation IDs before direct comparisons. |
| Baseline test assembly | A supplied recovery log reports both output files assembled for 1,732,544 Source 1 rows, with internal coverage, target-ID, uniqueness, and candidate-subset checks passing. It reports 88,591,494 candidates and 4,953,551 accepted links. | Treat this as a reported recovery checkpoint. The files were not independently downloaded and checked here; the organizer validator and portal status are still unverified. |
| Person A candidate implementation | Candidate generator, deterministic long-form output, and fixture tests are on the remote feature branch; this is not yet integrated into AJ's checkout. | Reuse that branch's code and tests after pinning the exact commit and exporting the tested settings. |
| Person A full-target retrieval | The current remote tracker reports the best completed comparable development result as 32,594 / 34,770 links (93.7417%) across 10,000 fixed queries searched against all training targets, 639,042 final pairs, oracle macro F0.5 0.977017, and about 20 GiB reported peak process RSS. Its final cap is 64 combined across S2 and S3 per Source 1; the first-stage shortlist is up to 64 per target source and final selection starts at 32 per source before redistributing unused quota. The 94.9928% figure is after that initial per-source ranking cut; raw proposal recall was not measured. | Start the next retrieval comparison from this broad-token run, with its exact commit/config and candidate budget recorded. Do not call the 10k-query run full-training recall. |
| Other retrieval experiments | The same tracker reports an isolated name-key variant at 90.6327% on those 10,000 queries, below its matched comparison, and a stable-hash gram run was pending at the last recorded update. Two full-training shards around 90.7% use an older configuration; aggregation and newer-config shards were unfinished. | Do not promote name keys alone. Recheck pending work, then use only results with a completed report and matching configuration. |
| Person A full-test candidates | A remote tracker entry says the cap-64 full-test long file passed candidate-ID, uniqueness, ordering, and source coverage checks for 110,336,665 pairs and 1,732,543 Source 1 IDs; one Source 1 needs an empty row at wide assembly. Its recorded path under D:\\temp was absent when checked for this plan. | Treat as a reported historical fallback only; recover and revalidate the actual file before relying on it. |
| Separate 99.84% report | A team screenshot reportedly showed 6,975 / 6,986 links on 2,000 queries, but the query membership, code, cap, candidate file, and label-blind process were not available in the checked branch. | Treat it as an experiment lead only. Reproduce on the fixed query IDs, full target pools, and the same candidate budget before considering it. |
| AJ diagnostics | Preprocessing and retrieval-diagnostic code/notebook files are present in the working tree, but their full diagnostic run is not verified in this plan. | Run them as controlled diagnostics; synthetic tests or notebook presence alone do not establish retrieval quality. |

The original notebook failure is also useful evidence: empty normalized strings caused DuckDB Jaccard to fail after millions of candidate pairs. The current path includes empty-string guards, uses the Kaggle-provided XGBoost rather than downgrading packages, and uses the validation-selected best iteration for inference. Keep these fixes.

References: [AJ audit](../experiments/aj_kaggle/DATA_AUDIT.md), [AJ run and recovery notes](../experiments/aj_kaggle/README.md), [reconciliation of team findings](../experiments/aj_kaggle/TEAM_FINDINGS_RECONCILIATION.md), and [Person A's live tracker](https://github.com/Atishay9828/Amazon_ML/blob/feat/data-candidates/docs/person-a/todo.md). Remote branch reports are attributed to that tracker, not independently rerun by this plan.

## What to borrow from top Kaggle solutions

The relevant precedent is not “use the biggest neural model.” In the Foursquare matching competition, a first-place team documented distinct candidate-generation, feature/model, and post-processing stages, with cross-validation and separate evaluation/submission training. A 7th-place Kaggle Competition Master writeup describes complementary retrieval signals, a LightGBM pair classifier, text normalization and edit/overlap similarities, validation folds, and chunked inference to manage memory. These are useful workflow patterns for our task.

We will adapt them as follows:

1. **Retrieval is its own measurable subsystem.** Combine useful name, address, exact-key, token, character-gram, and—only if cheaper channels leave measurable misses—semantic channels. Record which channel retrieved each pair.
2. **A reranker scores the retrieved pairs.** Begin with the existing pairwise XGBoost design and conventional similarity, number-agreement, missingness, and channel-hit features. Train Source 2 and Source 3 models separately on the two T4s. Consider a heavier text model only after error analysis shows a gap that simpler features do not address.
3. **Normalize for several views.** Retain original text and derive conservative character and token views. Do not let normalization erase numeric address information or convert blanks into broad blocks.
4. **Validate the whole output unit.** Split and score by Source 1 entity, include empty/singleton rows, and calculate the competition metric after converting pair scores into each entity's complete predicted match set.
5. **Build for restart and memory limits.** Use sparse/indexed retrieval, bounded batches, checkpoints, and a run manifest. The winning solutions process stages separately; we should not require all 10M-plus target records or all pair features to fit in RAM at once.
6. **Keep the final notebook reproducible.** Use a short configuration/preflight section, ordered experiment cells, deterministic IDs/seeds, explicit input/output locations, and a manifest. Keep reusable implementation in versioned Python source and generate or attach that source to Kaggle. A development notebook can be exploratory; the final inference notebook should be linear and restartable.

The source competitions had extra information and risks that do not transfer. In particular, the Foursquare task had latitude/longitude and other fields that our data does not have, and its first-place writeup discusses exploiting train/test overlap. That leaderboard gain is not evidence to copy; we will use neither location features nor leakage-dependent post-processing. These two writeups are precedents for a staged workflow, not proof that every Kaggle Grandmaster uses the same notebook structure. Sources: [first-place Kaggle solution](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution), [7th-place Kaggle Competition Master writeup](https://future-architect.github.io/articles/20220720a/), and its [solution repository](https://github.com/TakoiHirokazu/Kaggle-Foursquare-Location-Matching).

### Why start with 10,000 queries?

The 10,000 is the number of Source 1 queries used to compare retrieval settings; it is not the size of the target search. Each query is searched against the complete training Source 2 and Source 3 pools—10,320,219 target records in total. This keeps repeated experiments affordable and comparable while preserving the real search scale. It still does not establish performance for every training query or for test; the selected configuration must later run across all training Source 1 rows and all test Source 1 rows.

## End-to-end execution

| Step | Work and owner | Exit gate / saved evidence |
|---|---|---|
| 1. Reconcile the live state | AJ checks current local modifications, remote branches, current tracker updates, and Kaggle session artifacts. Person A supplies the exact current retrieval commit/config. Person B supplies the matcher branch, split, model, and logs; no Person B branch/result is verified in this checkout yet. | One evidence ledger separates local files, remote commits, supplied logs, and verified portal results. Refresh stale README/experiment statements. No uncommitted work is discarded. |
| 2. Freeze a shared interface | AJ, A, and B agree one schema for a unique long-form candidate pair, source IDs, retrieval-channel flags/scores, and the exact candidate set sent to scoring. The matcher joins the pair IDs back to the source rows and emits one score per candidate; it may not silently filter candidates. | Contract and a small schema example are checked in. Empty-candidate Source 1 IDs are preserved at wide-output assembly. Candidate caps are explicitly per target source or combined. |
| 3. Freeze evaluation samples | Keep A's 10,000-query retrieval-tuning set separate from model validation and holdout: its retrieval choices have already been tuned against those labels. Freeze disjoint Source 1 ID sets for model validation and a fresh blind holdout from the remaining rows before more experiments. The old 50k holdout was already read with a baseline, so it is not blind. Preserve every Source 1 truth list together. Keep the complete training S2/S3 target corpora searchable in every query fold; do not split target records by fold. The audit found no target ID repeated across different Source 1 truth lists. | Save the exact private ID lists and hashes, seed, country/singleton/link-count counts, and DuckDB version. The existing AJ split uses DuckDB hash values, which can vary by version; do not rely on recomputing a split when its IDs can be saved. Generate candidates without labels and join labels only for measurement/training. |
| 4. Benchmark retrieval fairly | A runs the current broad-token candidate generator and controlled alternatives on exactly the same 10k queries and all target records. Compare AJ retrieval at its native 80-pair maximum and normalize both systems to an equal 64-pair combined budget for the head-to-head; keep the per-source and intermediate limits visible. Vary one retrieval setting at a time. | For every run: true-link recall, complete-match-set coverage, oracle macro F0.5, candidates per query (mean/p95/max), cap saturation, runtime/RSS, and missed-link attribution by source, country, and blank-address slice. Select on oracle macro F0.5 and operational cost together. Aim toward 98% candidate-link recall, but never declare success on recall alone. |
| 5. Integrate and train the matcher | B (or AJ if no matching handoff is available) joins the selected candidate pairs to original records, creates pair features, labels from full training ground truth, and trains separate S2/S3 XGBoost models on T4 GPU 0/1. Use hard negatives from the actual retrieval output; IDs are joins only, not features. Preserve XGBoost already installed on Kaggle; do not force package downgrades. | Validation file contains one score for every candidate and a row for every Source 1 when scored wide. Log model version, feature list, best iteration, thresholds, seeds, training sample policy, and GPU assignment. |
| 6. Compare end-to-end and choose threshold | AJ/B evaluate the original model and candidate/model combinations on the same frozen Source 1 validation set. Select a threshold for macro F0.5 over all entities, including those with no candidates or no true links. Inspect false positives, missed links, singleton errors, and country/source/missing-address slices. | New integrated result beats the re-run baseline on the agreed metric or the team keeps the baseline. Freeze retrieval, features, model, and threshold before looking at the fresh holdout once. Do not tune after the holdout read. |
| 7. Run full training evidence | A applies the selected retrieval config to all 2,206,821 training Source 1 rows in resumable shards and aggregates exact link/query denominators. B trains final models on the permitted training rows and saves inference artifacts. Older shards from other configs do not count toward this result. | All shard identities and row/link counts reconcile; no gaps or overlaps; selected-config recall and resource totals are reported. |
| 8. Generate test predictions | Run every 1,732,544 test Source 1 record against the full test Source 2/3 pools, including France, in bounded/resumable partitions. Retrieve the final candidate set and score every candidate with the frozen model. | Manifest records commit/config, input hashes, installed package versions, thresholds, partition completion, model/candidate/output hashes, counts, and timing. France coverage is checked; France accuracy is not claimed. |
| 9. Assemble and validate outputs | AJ streams long-form candidates and accepted matches into both wide TSVs. Include every Source 1 ID exactly once in each file, even if its candidate or match list is empty. | Internal checks pass for full coverage, valid target IDs, uniqueness, and final-match subset of candidates. Then run the organizer validator and retain its actual PASS output. Internal checks do not substitute for it. |
| 10. Package and submit | AJ prepares the required runnable source/notebook, pinned or recorded environment, two TSVs, completed methodology, and reproducibility notes; privately archives models and detailed logs as required. Upload the correct file through the portal. | Record filename/hash, run/commit, upload time, actual portal status, and leaderboard score only after the portal shows them. Keep “assembled,” “validator PASS,” and “SCORED” as separate checkpoints. |

## Kaggle notebook layout we should use

Use two notebook roles:

- **Experiment notebook:** preflight and dataset checks; fixed sample/config; candidate retrieval comparison; recall/oracle report and error slices; selected candidate export. One hypothesis per run, with a comparison table.
- **Final T4 notebook:** (1) environment/GPU/input preflight, (2) load configuration and manifest, (3) validate data and split metadata, (4) train the two source-specific models, (5) score the frozen validation and choose or load the frozen thresholds, (6) run chunked full-test retrieval and scoring, (7) stream outputs and run internal checks, (8) save manifest and completion files.

Keep cells ordered and safe to rerun from a recorded checkpoint. Make progress, elapsed time, memory, row counts, and completed partitions visible. Keep the implementation in readable, version-controlled source; the Kaggle notebook should orchestrate those sources or contain a generated snapshot tied to a commit. Do not place data samples, tokens, or generated outputs in the public notebook/repository.

## Immediate work order

1. Refresh this plan from current branch/tracker evidence and mark which prior Kaggle outputs are actually available to download.
2. Agree on the disjoint ID sets, candidate-pair schema, normalized candidate budget, matcher input, and fresh holdout membership.
3. Run the existing AJ diagnostic notebook for its stated purpose, then re-run the baseline on the agreed model-validation IDs. Compare retrieval systems only when query IDs and target pools match.
4. Finish or check the pending Person A experiments; compare only completed artifacts at equal budgets. Start with broad-token and its miss categories; do not restart costly runs without checking saved checkpoints.
5. Connect the selected candidate set to the T4 matcher and run one complete validation pass.
6. Freeze the choice; then do full train shards, full test inference, output validation, package, and portal confirmation.

The repository currently contains uncommitted preprocessing diagnostics. Keep those changes intact while working this plan; this plan does not authorize replacing, cleaning, or publishing them.

## Main risks and controls

| Risk | Control |
|---|---|
| Candidate cut loses positives before the model | Measure each stage separately and retain miss attribution; compare recall and oracle score at the same pair budget. |
| Small samples are mistaken for small search | Print both query count and full target population in each report. Full-target 10k-query experiments are controlled development measurements, not full-query results. |
| Retrieval branches use incomparable splits or caps | Freeze query IDs and report per-source/combined caps and every intermediate cutoff. |
| Label leakage or optimistic validation | Generate candidates without labels; split by Source 1 and keep all of its truth together; use a fresh holdout once. |
| Memory spike or lost Kaggle session | Stream and checkpoint; write completion metadata last; save/download private outputs and manifests before the session is discarded. |
| France is out of training distribution | Preserve dynamic country routing and test coverage; state clearly that France quality is unmeasured. |
| Reported output is treated as organizer acceptance | Require actual organizer validator PASS and portal status evidence separately. |

## Definition of done

The selected pipeline is reproducible from the exact code/config and private official input; its held-out macro F0.5 is reported alongside candidate recall and retrieval oracle; full test processing covers all Source 1 rows and both target pools; both output files pass internal checks and the organizer validator; the methodology and package match the run manifest; and the portal shows the actual submitted/scored state. Until those artifacts exist, results remain baselines or reported checkpoints rather than a completed submission.
