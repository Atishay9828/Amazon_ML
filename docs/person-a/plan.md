# Person A — candidate generation

Owner: Mridul. Person B owns matching/evaluation; Atishay owns integration, packaging, and portal submissions. Deadline: 27 September 2026, 23:59 IST.

## Contract and boundaries

- Read the official train/test TSVs from `--data-root`; never commit raw data, labels, generated pairs, or credentials.
- Emit the **last** candidate set sent to Person B as a sorted long-form TSV: `source1_entity_id`, `candidate_entity_id`, `name_cosine`, `address_cosine`, `retrieval_channels`. A Source 1 record with no candidates has no long-form row. C makes its empty wide-format row.
- Keep IDs as opaque strings. Candidate IDs must be from Source 2/3 of the same split. Scores must be finite and within [0,1]. Country is an open-set attribute, never a hard filter.
- Own `src/data.py`, `src/normalization.py`, `src/candidates.py`, focused tests, and `docs/data_audit.md`. Do not edit B/C-owned implementation or shared root docs.

## Work sequence

1. Audit all files with explicit tab parsing: headers, malformed rows, ID uniqueness/prefix, blank fields, country counts, lengths, label coverage, cardinality, and cross-country true links. Preserve source hashes and commands in `docs/data_audit.md`.
2. Normalize consistently in train and test: Unicode NFKC, generic AnyAscii script transliteration, casefold, ampersands, punctuation/whitespace, common legal and address variants; retain meaningful digits and raw text. Blank values never form exact-match blocks. AnyAscii contains generic character mappings, not business identities.
3. Retrieve with exact normalized name/address, rare name/address tokens, and float32 character 3–5 gram TF-IDF name/address similarity. Use a bounded rare-gram inverted index to propose records, then compute sparse cosine on those records. Search Source 2 and Source 3 separately rather than making a Cartesian join. Union/deduplicate, score both text fields for each emitted pair, and cap only after retrieval.
4. Start with name K=32 and address K=16 per target source. Compare final total caps 16/32/64 on development groups. Measure link recall, complete true-set coverage, candidate mean/p95/total, oracle macro F0.5, runtime, and peak RAM. Prefer the smallest configuration within 0.002 oracle F0.5 of the best. Analyze misses; revisit retrieval if recall is below 98% rather than claiming it met the target.
5. Smoke test with invented fixtures and a local sample. Use a **private** Kaggle CPU environment for the full run if local resource limits are confirmed. A GPU is allowed by challenge rules; switch only after a measured implementation benefits. Keep the runnable CLI path-independent and record versions/resources.
6. Hand Person B the exact train/test candidate tables and metrics. Push A-owned commits to `feat/data-candidates` and hand Atishay source, audit, selected settings, dependencies, and reproducible commands directly; Atishay assembles and validates the official wide outputs.

## Evidence and decisions

Use `docs/person-a/todo.md` as the work log. Mark a task complete only with a dated command, artifact, metric, or commit. Attribute actions to their actual owner. Person B's connected-component development/holdout split is authoritative when available; do not tune on its untouched holdout or public leaderboard.

## Recall improvement decision sequence (26 Sep 2026)

The original bounded broad-token development file retrieved 32,594/34,770 true links at a final cap of 64 (93.7417%). Re-selecting its controlled 2,048-row-per-source stage with the implemented balanced final rule kept 32,936 links (94.7253%) at the same cap, but this version has not been run across full train/test. Reaching 98% on these same labels would require at least 34,075 links, or 1,139 more than that best offline selection. This is a candidate-recall target, not an achieved result or leaderboard score.

1. The broad-token run measured 94.9928% after its first 64-row-per-source ranking cut and 93.7417% at final cap 64. **Staged recall is after `_rank` has already cut each source; it is not raw proposal recall.** Its controlled 2,048-row stage topped out at 95.6744% with only 14 of 20,000 source groups at the limit, confirming that these proposal channels cannot reach 98% by ranking alone.
2. Compare isolated proposal changes on the same seeded 10,000 Source 1 IDs and full train targets. Stable-hash grams gained 59 staged links; larger hash space and minimum index DF did not materially help. Name keys, wider word probes, wider character-gram budgets, and same-source second hop are running. Measure staged/final recall, candidate volume, runtime, and **container** peak memory. Keep final cap fixed unless the gain justifies Person B's larger scoring load.
3. Use the best measured proposal configuration and balanced or learned final selection. The separate 50,000-entity broad-token/balanced holdout recovered 94.2862% of true links, close to its 94.4492% development result, but it does not validate untested variants. Do not tune new settings from holdout labels. Recheck the chosen candidate set with Person B's actual matcher when available.
4. Generate full train and test tables with identical chosen settings, verify the five-column contract and candidate coverage, and hand exact pair tables and reproduction commands to B/C. Person B should evaluate matching F0.5, including singleton behavior, before C makes leaderboard and package submissions. A high retrieval recall alone cannot establish a 0.98 matching score.
