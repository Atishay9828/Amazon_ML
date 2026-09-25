# Person B — Evaluation and Matching

This is Person B's work brief for the Amazon ML Challenge 2026. AJ is Person C and owns integration, GitHub merges, documentation, and submissions. Person A owns data loading, normalization, and candidate generation. Read the [shared challenge context](../../README.md) and [team plan](../../PLAN.md) before implementing challenge behavior.

## Mission and boundary

Implement trustworthy validation and a model that scores every pair Person A generates. For each Source 1 entity, the final decision may include zero, one, or many Source 2/3 IDs. Optimize macro F0.5, calculated per Source 1 entity and averaged across all entities. Candidate coverage is a separate ceiling measured by Person A.

Inspect the current checkout first. If official data is not yet available, start the scorer, split logic, feature functions, and focused invented fixtures. Do not invent measured results.

## Git and file ownership

Create branch **feat/matcher-evaluation** from current origin/main. Own only:

- code/business_entity_resolution/src/split.py
- code/business_entity_resolution/src/metrics.py
- code/business_entity_resolution/src/features.py
- code/business_entity_resolution/src/model.py
- code/business_entity_resolution/src/train.py and score.py
- A thin optional training notebook and focused tests for your modules
- docs/model_report.md

Do not edit Person A's loader, normalizer, or candidate generator. Do not edit the root README, PLAN.md, requirements, output assembler, or methodology; AJ owns those. Send exact dependency names and working versions to AJ. Push your branch and open a pull request to main; AJ reviews and merges. Do not upload to the portal or merge your own pull request.

Never commit official records, ground-truth rows, generated pairs, predictions, model artifacts, cloud credentials, or submission ZIP files to this public repository.

## Fixed pair contract

Person A writes a long-form candidate TSV with exactly these columns:

    source1_entity_id    candidate_entity_id    name_cosine    address_cosine    retrieval_channels

The displayed spaces are for readability; both files use actual tab separators. There is one row per candidate pair; no row represents an empty candidate list. Join candidate IDs to raw source name, address, and country. Score every candidate row and write the same pair set as:

    source1_entity_id    candidate_entity_id    score

Preserve exact IDs and pair cardinality. The score must be finite and numeric. No hidden filtering is allowed between candidate input and scoring. AJ applies the selected threshold and assembles the two required wide-format output files. Deliver the model artifact, threshold, feature-order metadata, and exact commands to AJ. Commands accept paths as arguments, not hard-coded machine or notebook paths.

## Evaluation — implement this first

1. **Exact scorer.** Convert true and predicted ID lists to sets and score each Source 1 entity. Both empty scores 1.0; exactly one empty scores 0.0. Otherwise calculate precision, recall, and F0.5 = (1.25 × precision × recall) / (0.25 × precision + recall). Return the arithmetic mean over all Source 1 entities, including singletons. Two correct IDs plus one false ID against two true IDs should score approximately 0.714. Pair accuracy, ROC AUC, and micro F0.5 are diagnostics, not the selection metric.
2. **Leakage-aware split.** Build connected components of the positive-link graph spanning Source 1 IDs and their linked Source 2/3 IDs. A shared target cannot connect a development Source 1 entity to a holdout entity. With seed 2026, place 20% of groups in an untouched holdout and create three development folds. Balance country and zero/one/many-match categories where component sizes permit; never break a component for balance. Provide a command that regenerates a local split manifest with source1_entity_id, partition (dev or holdout), and fold (0–2 for dev, blank for holdout). Keep record IDs and the manifest out of public GitHub.
3. **Baselines.** On the real data, calculate an all-empty baseline and a conservative exact normalized name/address baseline with this same scorer and split. Record singleton prevalence and match-count distribution. Do not present an all-empty score as evidence of good matching.

## Matching model

4. **Features.** Use Person A's candidate rows and source records. Build null-aware comparisons: normalized name/address equality, token overlap and containment, character or TF-IDF cosine, edit or token-sort similarity, numeric address-token agreement/mismatch, string-length ratios, missing-field flags, target-source indicator, country equality, and country missingness. Country labels remain open; never one-hot only US and India. IDs and numeric ID fragments are join keys, never predictive features. The feature pipeline must work for unseen France records.
5. **Training examples.** Label positive candidate pairs from complete training ground-truth lists. Other generated candidates are negatives, including look-alikes and singleton candidates. Train on retrieval-stage negatives rather than random noncandidate Cartesian pairs. If the training candidate table exceeds two million pairs, retain every positive and sample negatives reproducibly across Source 1 entities and retrieval ranks. Validation and holdout always score all candidates.
6. **First learned model.** Begin with a small CPU LightGBM binary classifier, seed 2026, and saved configuration and feature order. The project has an [MIT license](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE). Compare only a limited, logged set of configurations after the scorer and split are trusted. Report macro F0.5, singleton accuracy, predicted match-count distribution, false merges, missed links, and results by target source and training country. Avoid a large model search before resolving data and candidate quality.
7. **Threshold and holdout discipline.** Generate out-of-fold scores on development groups. Sweep thresholds and choose the one maximizing macro F0.5 across Source 1 entities; choose the higher threshold on an exact tie. Allow multiple accepted links, with no global one-to-one assignment. Freeze candidate configuration, feature set, model, and threshold before scoring the untouched holdout once. Do not retune from the holdout or public leaderboard.

## T4 option

If three-fold CPU model training takes more than 30 minutes or training candidates exceed two million pairs, benchmark an [Apache 2.0 XGBoost](https://github.com/dmlc/xgboost/blob/master/LICENSE) histogram classifier on an available Colab or Kaggle T4 using the same feature table, folds, and metric. Use device=cuda and tree_method=hist when compatible with the pinned runtime; see the [official GPU guidance](https://xgboost.readthedocs.io/en/stable/gpu/). Adopt it if development macro F0.5 improves, or stays within 0.002 of LightGBM while training at least twice as fast. Otherwise retain LightGBM. Report compatibility failures honestly and continue with CPU.

The notebook only installs pinned dependencies, accesses official data privately, invokes repository source commands, and saves a model and run metadata. There is no notebook-only feature or training logic. Save checkpoints because cloud sessions can end. The selected model must support CPU inference in the final package.

## Checks and handoff

Use focused invented cases for singleton scoring, false merges, multi-match sets, graph split isolation, unseen country strings, missing text, stable feature order, and preservation of every scored candidate row. Integrate with Person A's candidate file once available; do not copy its production loader merely to unblock a fixture.

Open a pull request with: exact commands; dependency and model versions for AJ to pin; split logic and evidence that connected components stay together; real baseline/model scores if available; threshold; runtime and memory findings; major false-merge and miss categories; and any limitation from missing data. AJ owns final wide TSVs, official validator, ZIP, and portal upload.
