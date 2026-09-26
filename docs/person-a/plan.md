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

The current best measured 10,000-entity development run is 32,028/34,770 true links at a final cap of 64: 92.1139% link recall. Its staged per-source shortlists contain 32,390 links (93.1550%). Reaching 98% on these same labels would require at least 34,075 links, or 2,047 more than the current final result. This is an optimization target, not an achieved result or a leaderboard score.

1. Compare the running bounded broad-token run against the same seeded Source 1 IDs and full train Source 2/3 pools. Report staged and final recall, not only the latter; higher final recall may conceal a still-low proposal ceiling. Keep the 64-candidate final cap for the direct comparison.
2. If the proposal ceiling stays below 98%, inspect its missed pairs by shared name/address words, missing fields, target-source saturation, and country. Test one bounded retrieval change at a time: more word-pair postings, more character gram probes, a deliberately controlled single-word channel, or exact name keys for reordered words and joined website names. Record candidate volume, runtime, and RAM with every recall result.
3. If proposal recall is high but final recall is low, compare `--stage-cap 64` against 128/256 per target source, then improve shortlist ranking and source allocation while holding the final cap fixed. Raise the final cap only if the measured recall gain justifies the much larger scorer input and submission artifact.
4. Apply the chosen configuration once to the untouched 50,000-entity holdout, then generate full train and test tables with identical settings. Person B should evaluate the resulting matcher F0.5, including singleton behavior, before C makes leaderboard and package submissions. A high retrieval recall alone cannot establish a 0.98 F0.5 score.
