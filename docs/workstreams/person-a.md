# Person A — Data Audit and Candidate Generation

This is Person A's work brief for the Amazon ML Challenge 2026. AJ is Person C and owns integration, GitHub merges, documentation, and submissions. Person B owns evaluation and matching. Read the [shared challenge context](../../README.md) and [team plan](../../PLAN.md) before making assumptions about the data or rules.

## Mission and boundary

For every Source 1 record, produce a high-recall, bounded set of plausible Source 2 and Source 3 records. Source 1 is deduplicated. A Source 1 entity may have zero, one, or many true matches. Your candidate table is the exact set of pairs that Person B later scores; it is not an earlier, broader block.

The official dataset may not yet be available. Inspect the current checkout and begin with small invented fixtures if necessary. Do not report invented record counts, recall, or challenge scores.

## Git and file ownership

Create branch **feat/data-candidates** from current origin/main. Own only:

- code/business_entity_resolution/src/data.py
- code/business_entity_resolution/src/normalization.py
- code/business_entity_resolution/src/candidates.py
- Focused tests for those modules
- docs/data_audit.md

Do not edit Person B's split, metric, feature, or model modules. Do not edit the root README, PLAN.md, requirements, output assembly, or methodology; AJ owns those. Send dependency names and working versions to AJ in your pull request. Push your branch and open a pull request to main; AJ reviews and merges. Do not upload to the challenge portal.

Never commit official data, labels, generated candidate tables, predictions, model artifacts, cloud credentials, or the submission ZIP to this public repository.

## Fixed output contract

Write a long-form TSV with this header and column order:

    source1_entity_id    candidate_entity_id    name_cosine    address_cosine    retrieval_channels

- The displayed spaces are for readability; write actual tab separators.
- IDs are strings with exact S1-/S2-/S3- prefixes. The candidate ID must exist in the same split's Source 2 or Source 3 file.
- name_cosine and address_cosine are finite values in [0, 1]. Write 0 when a retrieval channel did not score the pair.
- retrieval_channels is a deterministic, pipe-separated list, for example exact_name|name_char. It is for auditability.
- Emit one row per unique pair, sorted by Source 1 ID and candidate ID. An entity with no candidates has no long-form row; AJ's assembler creates its empty wide-format row.
- Apply any cheap pair filtering before writing this file. Person B must score every row in it.

Provide a CLI accepting --data-root, --split train|test, and --out. All paths come from arguments, never a local, Colab, or Kaggle absolute path.

## Work sequence

1. **Build reliable loading.** Parse source TSVs with an explicit tab delimiter. Preserve IDs and text as strings and empty fields as empty strings. Validate headers, duplicate IDs, source prefixes, and malformed rows. Return raw business name, address, and country for later features. Treat country as an open string label, not an enumeration of US and India.
2. **Audit the real files.** Once they arrive, report row counts, duplicate and missing IDs, field missingness, country values, name/address lengths, training-label coverage, nonexistent referenced IDs, and zero/one/many-match distribution. Check whether true links ever have different country labels. Separate observed facts from hypotheses in docs/data_audit.md. Escalate invalid labels to AJ; do not discard them silently.
3. **Normalize conservatively.** Provide Unicode normalization, case folding, punctuation/whitespace handling, and comparison variants for ampersands, common legal suffixes, and address abbreviations. Preserve original strings and meaningful digits. Never allow two blank values to form a broad exact-match block. Use the same transformations in train and test.
4. **Build multiple retrieval channels.** Start with exact normalized name, informative exact normalized address, rare name-token overlap, and character 3–5 gram TF-IDF similarity for name and address. Search Source 2 and Source 3 separately so one source cannot consume the other's top-K budget. Use batched sparse top-N retrieval rather than a full Cartesian join. Do not hard-filter by country; France is unseen in training but present in test.
5. **Start with explicit budgets.** Initial values are name top-K 40 and address top-K 20 per target source, unioned and deduplicated, with a 500-pair cap per Source 1. If the training audit finds an entity with more than 250 true links, raise the cap to at least twice the observed maximum. Make rankings and ties deterministic.
6. **Tune only on development groups.** Person B owns the deterministic connected-component split generator. You can build and smoke-check retrieval before it lands. After it lands, compare name K in {20, 40, 80} and address K in {10, 20, 40} on development groups only. Do not use its untouched holdout or public leaderboard scores to choose retrieval settings.
7. **Measure retrieval separately from matching.** Report true-link recall, proportion of Source 1 entities whose full true match set is present, candidate count distribution, total pairs, runtime, memory where practical, and oracle macro F0.5 when Person B's scorer is available. Categorize missed links by observed cause: weak/missing name, weak/missing address, DBA name, transliteration, typo, block overflow, and other. If true-link recall is below 98%, investigate the largest recoverable category and try a targeted retrieval change before proposing a frozen setup. Report unresolved misses honestly.

## Checks and handoff

Use small invented fixtures for tabs with commas inside addresses, empty values, Unicode, duplicate suppression, valid target IDs, deterministic ordering, no-candidate entities, and France records. On real data, profile before running an expensive job. Candidate generation is primarily sparse CPU text work; a T4 is optional only if a measured implementation uses it. The source command must stay portable.

Open a pull request with: changed files, exact commands, candidate schema, dependencies for AJ to pin, observed audit findings if data is available, retrieval metrics and resource use if measurable, unresolved misses, and any block caused by missing data. Do not claim final matching quality; Person B owns that measurement.
