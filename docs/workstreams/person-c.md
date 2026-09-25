# Person C — AJ, Integration and Release

**Owner:** AJ (@Atishay9828). You coordinate the Amazon ML Challenge 2026 team and are the only person who merges team work into main or uploads challenge submissions. Person A owns data and candidates; Person B owns evaluation and matching. The [shared context](../../README.md), [execution plan](../../PLAN.md), and [experiment log](../experiments.md) are your coordination references.

This brief describes responsibilities and acceptance gates; it does not claim the dataset, model, or outputs already exist.

## Own the shared contract and repository

Own the root README and PLAN.md, docs/workstreams/, docs/experiments.md, .gitignore, pinned dependency/environment files, the eventual output assembler, package builder, and methodology document. A and B own their assigned modules and technical reports. Request fixes through their pull requests or issues instead of directly editing their files. Keep a clear branch and review trail; merge only after the agreed handoff passes.

The internal candidate handoff has one row per unique pair with source1_entity_id, candidate_entity_id, name_cosine, address_cosine, and retrieval_channels. Person B returns one score for every candidate pair, preserving the pair set. You own the conversion of these long-form tables into the two required wide-format TSVs. If either side needs a schema change, approve and document it before both teams implement different versions.

Create and maintain three work items: data/candidates for A, evaluation/matcher for B, and integration/submission for yourself. Require each pull request to state its exact commands, real measurements where available, dependencies, resource use, known limitations, and what remains blocked. Record the merged commit for every integrated run.

## Start before the dataset arrives

1. Maintain a Python 3.12 environment definition. Pin working package versions after A and B report them; do not guess versions in advance. Confirm that the license of the selected final model is MIT or Apache 2.0 and document the exact model version.
2. Keep official data, generated predictions, model artifacts, and ZIP files out of this public repository. The root .gitignore protects common paths, but also inspect staged files before every push.
3. Define the output assembler's interface: inputs are test Source 1 IDs, A's final candidate TSV, B's scored TSV, and B's selected threshold. Outputs are output/candidate_pairs.tsv and output/matching_results.tsv. It must reject pair-set mismatches, invalid/non-finite scores, duplicate IDs, and missing Source 1 coverage; it must emit an empty list where appropriate.
4. Prepare the final package layout from the challenge context. Keep the development source in code/business_entity_resolution/src/ so copying it to the required ZIP is straightforward. Maintain exact run instructions and a filled methodology document based on actual work rather than a generic description.

## On official data arrival

1. Obtain the challenge archive through the official portal. Store it locally or in access-controlled cloud storage, never in the public GitHub repository. Record its SHA-256, layout, acquisition time, and the source of truth for all teammates. Do not use a shared simultaneous portal login.
2. Confirm A's audit checks source and label integrity before anyone reports a score. Confirm B's split keeps positive-link components together and that its untouched holdout is not used for retrieval or threshold tuning.
3. Resolve handoff problems by comparing actual headers, pair counts, ID sets, and command output. The exact candidate set that B scores must be the set you serialize into candidate_pairs.tsv. Do not permit an unreported filtering pass between them.
4. Maintain [the experiment log](../experiments.md). For each run record owner, timestamp, Git commit, dataset checksum, split seed, candidate configuration, model/features, threshold, candidate recall and volume, macro F0.5, singleton behavior, runtime, memory, decision, and any portal result. Mark unavailable measurements as unavailable; do not infer them.

## Integration, validation, and submission

Run the end-to-end pipeline from a clean Python 3.12 environment using only the provided training/test data and the committed code plus packaged model artifact. Check deterministic row coverage and ordering. Validate that all matched IDs are unique Source 2/3 test IDs, every matched ID appears among the same Source 1 entity's candidates, and France Source 1 records remain present. Run the supplied utils/validate_submission.py and require PASS before an upload. The official validator checks format, not F0.5.

Only you perform portal uploads. The event instructions allow at most five per team per day. Give every submitted matching_results.tsv a run ID, commit SHA, file hash, validator result, local validation metrics, upload time, portal status, and public score when available. Confirm a SCORED status rather than treating a clicked upload as success. Use public leaderboard feedback as a sanity signal; keep model selection based on local labeled validation.

Assemble one final ZIP with output/matching_results.tsv, output/candidate_pairs.tsv, code/business_entity_resolution/src/, exact code README, pinned requirements or equivalent environment file, saved selected model, and filled Documentation_template.md. Include a concise 1–2 page summary if needed for the separate event instruction while retaining the full methodology template required by the challenge statement. Check that the package reproduces both output TSVs from the supplied data.

## Time checkpoints

- **25 September:** Publish shared plan and workstreams; obtain data; see A's audit and B's scorer/split; integrate the first candidate and scoring run. Aim for one valid submission if data arrives in time.
- **26 September:** Review measured retrieval and matching errors, merge focused improvements, and keep the experiment log current. Limit uploads to deliberate variants.
- **27 September by 12:00 IST:** Freeze candidate configuration, model, features, and threshold. By 18:00 IST finish retraining, output generation, official validation, ZIP, and methodology. Aim for final upload by 21:00 IST, leaving a buffer before 23:59 IST.

## Completion evidence

Your closeout states which Git commit and data checksum generated the two output hashes, the validator result, the exact package contents, model license, local macro F0.5 and its split, and the portal status. Clearly distinguish local validation, public leaderboard score, and unrevealed private leaderboard score. Do not claim success for a step without its evidence.
