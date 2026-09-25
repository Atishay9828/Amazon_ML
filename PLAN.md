# Amazon ML Challenge 2026 — Three-Person Execution Plan

This is the team's working plan, not a claim that the method has been implemented or scored. The [challenge context](README.md) records the supplied rules and known wording conflicts. The official dataset is not yet present in this repository. We will revise data-dependent choices only after recording the evidence and decision in [the experiment log](docs/experiments.md).

## Goal and non-negotiable contracts

For each Source 1 test entity, find zero, one, or many matching Source 2 and Source 3 IDs. Generate a candidate set, score exactly those candidates, and produce the two required tab-separated files: output/matching_results.tsv and output/candidate_pairs.tsv. Every Source 1 test entity gets one row in each file; a no-match or no-candidate list is empty. Every final match must occur in that entity's candidate list. Do not impose a global one-to-one assignment.

Select methods using macro F0.5 calculated per Source 1 entity, including singletons. The official validator checks format, while local validation measures matching quality. Never use external business databases, entity lookup, geocoding, internet data augmentation, or a pretrained entity lookup.

## People, ownership, and GitHub workflow

| Person | Owns | Primary deliverable |
|---|---|---|
| A — data and retrieval | Source loading, normalization, data audit, candidate generation | Final long-form candidate table and blocking quality report |
| B — evaluation and matching | Challenge scorer, leakage-aware split, features, pair model, threshold, optional T4 notebook | Scored candidate table, model artifact, frozen validation report |
| **C — AJ (@Atishay9828)** | Shared interfaces, pull request reviews and merges, experiment records, pinned environment, output assembly, validation, package, portal uploads | Reviewed integrated pipeline, both required TSVs, reproducible ZIP, submission history |

A owns data.py, normalization.py, and candidates.py under code/business_entity_resolution/src/, plus docs/data_audit.md. B owns split.py, metrics.py, features.py, model.py, train.py, and score.py under the same source directory, plus docs/model_report.md and the optional training notebook. C owns the root README and PLAN.md, dependency and packaging files, docs/experiments.md, docs/workstreams/, and the output assembly entry point. Each person edits their own area on a separate branch. A or B requests an interface change from its owner rather than editing the other person's files.

Use branches feat/data-candidates and feat/matcher-evaluation for A and B. Open pull requests to main with commands, results, dependencies, and limitations. AJ reviews and merges them; only AJ uploads to the challenge portal. This initial documentation change is itself a pull request for AJ to merge. All work must be traceable to a Git commit. Keep the official dataset, row-level predictions, saved models, and submission ZIP out of this public repository.

### Fixed integration handoff

A writes the exact candidate set fed to the model as a long-form TSV with these columns, in order:

    source1_entity_id    candidate_entity_id    name_cosine    address_cosine    retrieval_channels

The displayed spaces separate column names for readability; the file delimiter is a tab. There is one unique row per pair. IDs are strings; the candidate ID must exist in that split's Source 2 or Source 3 file. Scores are finite values from 0 to 1; retrieval_channels is a deterministic pipe-separated list. A Source 1 entity with no candidates has no long-form row. A sorts by Source 1 ID and candidate ID.

B scores every row of that candidate file and writes the same pair set as:

    source1_entity_id    candidate_entity_id    score

This file also uses tabs. B must not silently remove candidates before scoring. C assembles the required wide TSVs from these two long-form files, with one row per Source 1 ID and a stable ordering of comma-separated target IDs. Any cheap pair filter belongs before A writes its final candidate table.

## Technical sequence and decision gates

1. **Acquire and audit.** C obtains the official archive through the portal and records its SHA-256. A checks explicit tab parsing, required columns, ID prefixes and uniqueness, missingness, country values, label coverage, invalid references, and match cardinality. Check cross-country positive links. Report bad source data instead of dropping it silently. Official data stays local or in private cloud storage.
2. **Freeze validation.** B implements the supplied macro F0.5 formula, including 1.0 for an empty prediction on a true singleton and 0.0 for a nonempty prediction on one. Create positive-link connected components before splitting so linked records cannot straddle development and holdout. With seed 2026, reserve 20% of components as untouched holdout and use three folds on the remainder. Balance country and zero/one/many-match groups where component sizes permit. Record all-empty and conservative exact-match baselines.
3. **Build candidate retrieval.** A retains raw text and creates conservative normalized forms. Start with exact normalized keys, rare name-token overlap, and batched character 3–5 gram TF-IDF retrieval for name and address, searching Source 2 and Source 3 separately. Never exclude unseen country labels. Avoid materializing a Cartesian product. Begin with name top-K 40, address top-K 20 per target source and a 500-pair Source 1 cap; raise that cap if an observed training entity has more than 250 true links.
4. **Measure candidate quality.** On development groups, compare name K in {20, 40, 80} and address K in {10, 20, 40}. Report true-link recall, complete-match-set coverage per Source 1, candidate volume, runtime, and oracle macro F0.5. Pick the highest oracle score, breaking differences below 0.002 by lower candidate volume. If true-link recall remains below 98%, inspect the largest recoverable miss category and revise retrieval before freezing it. Holdout labels do not drive these choices.
5. **Train the matcher.** B labels candidate pairs from complete training ground truth and trains on retrieval-stage negatives, including look-alikes and singleton candidates. Use name/address token, character, and edit similarities; numeric address agreement; missing-field flags; source indicator; and country equality/missingness. IDs are join keys, never predictive features. Begin with a CPU LightGBM binary classifier, whose project license is [MIT](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE). Validation uses all candidates, even if training negatives must be sampled for memory.
6. **Choose the decision threshold.** Use out-of-fold development scores to maximize macro F0.5; on an exact score tie select the higher threshold. Keep multiple accepted links. Review false merges, missed links, blocking misses, singletons, and source/country slices. Freeze retrieval, features, model, and threshold before evaluating the untouched holdout once. France has no labeled training examples: verify its test coverage without claiming a measured France score.
7. **Use cloud GPU only for a measured bottleneck.** The repository code remains the source of truth; a Colab or Kaggle notebook only installs pinned dependencies, mounts private official data, invokes source commands, and saves run metadata. If three-fold CPU model training exceeds 30 minutes or training candidates exceed two million pairs, B benchmarks [Apache 2.0 XGBoost](https://github.com/dmlc/xgboost/blob/master/LICENSE) on an available T4 with the same features and folds. Use its documented [CUDA histogram configuration](https://xgboost.readthedocs.io/en/stable/gpu/) if compatible. Adopt it if development macro F0.5 improves, or stays within 0.002 while training at least twice as fast. Otherwise retain LightGBM. The final package supports CPU inference from the selected saved model.
8. **Build outputs and package.** C retrains the selected model on all labeled training data, generates test candidates and scores, assembles both output TSVs, runs the supplied validator, and prepares the required ZIP. Include runnable source, pinned environment, exact run instructions, saved model, and the filled Documentation_template.md. The methodology reports actual experiments, candidate strategy, feature/model choices, license, and limitations.

## Three-day checkpoints

- **25 September:** Publish plan and interfaces; obtain data; audit; finish scorer, split, first candidates, baselines, and an integrated run. Aim for one validator-passing leaderboard submission by evening if the data arrives in time.
- **26 September:** A tackles measured candidate misses; B tackles false merges and feature/model ablations. Run the T4 benchmark only if its gate is met. C records runs and submits no more than three hypothesis-led variants.
- **27 September:** Freeze the selected configuration by 12:00 IST. Retrain, generate full test outputs, reproduce the documented run, complete the ZIP and methodology, and pass the official validator by 18:00 IST. Aim for final portal upload by 21:00 IST, ahead of the 23:59 deadline.

AJ tracks the maximum of five portal submissions per team per day and records each filename, commit, portal status, and score. Use the public leaderboard as a sanity check; make model decisions from local labeled validation. Keep enough time for a submission to reach SCORED status and for package review.

## Acceptance checks

- Parsing preserves commas in addresses and ID lists while using tabs between columns.
- The scorer handles empty/empty, false singleton links, misses, false merges, and multiple matches exactly as specified.
- The scored pairs equal the candidate pairs; every final match is a candidate.
- Both test TSVs cover every Source 1 ID exactly once; listed IDs are unique, exist in test Source 2/3, and include any France records rather than filtering them out.
- The official validator prints PASS. A clean Python 3.12 environment with pinned dependencies and documented commands regenerates both outputs from the supplied files.
- The final archive contains the required code, outputs, and methodology. No raw challenge records, generated predictions, or model artifacts are committed to the public repository.
