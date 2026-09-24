# Amazon ML Challenge 2026 — Shared Context

**Purpose:** This README is the team’s shared reference for the challenge statement, organizer instructions, examples, constraints, known ambiguities, and our engineering workflow.

**Status:** Problem assimilation and process planning only. No blocking method, features, model, thresholds, or implementation have been selected.

**Source basis:** Participant-provided problem statement screenshots, event-instruction screenshots, and screenshots of the challenge video. This file is a structured summary; where the supplied materials disagree or are unclear, the issue is called out rather than silently resolved.

## 1. User request and source instructions

The current team objective is to understand the challenge and plan a rigorous ML-engineering process before designing a solution. The instructions quoted below are challenge rules addressed to participants. The challenge’s own “tips” and illustrative diagrams are identified as such; they are not additional hard requirements or decisions already made by the team.

## 2. Challenge objective

Business identity data comes from three independent data sources. Their records contain partial, noisy, or inconsistent information and do not share a common cross-source identifier.

- Source 1 is the deduplicated reference source.
- For each Source 1 record, find every matching record in Source 2 and/or Source 3.
- A Source 1 record may have zero, one, or many matches.
- The task is entity resolution: determine which records describe the same real-world business.
- There is no separate source column. The record’s file and entity ID prefix (S1-, S2-, S3-) indicate its source.

The challenge does not state a global one-to-one matching constraint. Do not introduce one as an assumption without evidence or organizer clarification.

## 3. Input schema and data variation

Each source record file contains:

| Field | Meaning |
|---|---|
| entity_id | Unique record identifier; prefix indicates Source 1, Source 2, or Source 3. |
| business_name | Business name. |
| business_address | Business address. |
| country | Country label for the record. |

All challenge files are tab-separated .tsv files. Commas can occur in addresses and in comma-separated ID lists, so files must be read with an explicit tab delimiter.

The written problem statement says training contains US and India; test also contains France, which is absent from training. Country is an open set of string labels. Do not hard-code or filter the pipeline to only US and India, and do not exclude France test records.

Expected name variation includes abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix inconsistencies, DBA/trade names, punctuation changes (such as & versus “and”), word-order changes, typos, and translations.

Expected address variation includes abbreviations (Rd/Road, St/Street), transliteration, missing components (such as PIN code or state), landmark references (such as “Near BI ATM”), municipal numbering formats, and component reordering.

## 4. Data files

| File | Contents |
|---|---|
| dataset/train/train_source1.tsv | Source 1 training records; deduplicated reference source. |
| dataset/train/train_source2.tsv | Source 2 training records. |
| dataset/train/train_source3.tsv | Source 3 training records. |
| dataset/train/train_ground_truth.tsv | Training labels for Source 1 entities. |
| dataset/test/test_source1.tsv | Source 1 test entities; every row needs a prediction. |
| dataset/test/test_source2.tsv | Source 2 test records. |
| dataset/test/test_source3.tsv | Source 3 test records. |

Training files have labels. Test files have no labels. The provided validation guidance is to hold out a split of the training data and score it with the stated F₀.₅ metric.

## 5. Ground truth

train_ground_truth.tsv has two columns:

- source1_entity_id: a Source 1 record ID.
- matched_entity_ids: comma-separated matching Source 2 and/or Source 3 IDs; empty when there are no matches.

Ground truth is one row per Source 1 entity. Each row contains that entity’s complete match set, not a separate row for each individual link.

## 6. Video examples and what they establish

The video frames are illustrative and explain the task; the example records and IDs are not identified as actual dataset rows.

### Sign-up and available attributes

A sign-up illustration shows Acme Robotics Inc at 500 Market St, San Jose, CA. Phone and email are shown as real-world fields, with a note that the challenge example uses name and address. The written record schema also includes entity_id and country. The intended role of country in matching is listed under open questions.

### Three vendor representations

One illustrative business appears as:

- Source 1, reference/deduplicated: Acme Robotics Inc.; 500 Market St, San Jose.
- Source 2, vendor A: Acme Robotics Incorporated; 500 Market Street, San Jose CA.
- Source 3, vendor B: Acme Robotics; Near City Hall, San Jose.

The slide emphasizes that the records come from different data vendors and have no shared ID.

### True match versus look-alike

A diagram links the Source 1 Acme Robotics record to an Acme Robotics Incorporated record from Source 2 and an Acme Robotics, City Hall record from Source 3. It shows Acme Bakery LLC as a Source 2 look-alike that should not be linked. The diagram reiterates that a Source 1 entity can have zero, one, or many matches.

### Candidate generation versus final matches

A blocking illustration uses an example S1-732914 Acme Robotics record. Its candidate list contains S2-11820, S2-54021, S3-905477, and S3-63118. The final matching illustration keeps S2-11820 and S3-905477.

The diagram describes an example block based on a cheap name-and-address key, which groups similar names or shared addresses and can include both matches and look-alikes. This is an explanation of candidate generation, not a mandated blocking algorithm.

The diagram clarifies the relationship between the required files: candidate_pairs.tsv records the last candidate set fed to the matching model; matching_results.tsv contains the final selected matches. Every final match must have been in that candidate set.

## 7. Required outputs and validation

Both output files go in output/ inside the final package.

### matching_results.tsv

This is the only file scored on the leaderboard and the file uploaded to the Portal. Columns:

- source1_entity_id
- matched_entity_ids — comma-separated Source 2 and/or Source 3 test IDs.

There must be exactly one row for every Source 1 test record. Leave the match field empty for no match. Do not repeat an ID within a list. Match IDs must exist in test data and must be from Source 2 or Source 3; Source 1 self-matches are prohibited.

### candidate_pairs.tsv

Columns:

- source1_entity_id
- candidate_entity_ids — comma-separated Source 2 and/or Source 3 test IDs.

Include one row per Source 1 test entity, including an empty list when blocking returns no candidates. IDs must be valid Source 2 or Source 3 test IDs, without duplicates. This must be the last candidate list immediately before model scoring, not an earlier blocking result. Every final match must appear in the corresponding candidate list.

This file is not leaderboard-scored. It is used to analyze candidate coverage/recall ceiling, reduction ratio, and pipeline behavior.

### Validator

The supplied standard-library-only helper is utils/validate_submission.py. Run from student_resource/:

python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

It prints PASS and exits 0 if the files satisfy the rules, or prints numbered issues and exits 1. It reads the output files and test sources; it does not calculate the challenge score.

## 8. Evaluation and leaderboards

The metric is macro-averaged F₀.₅, calculated per Source 1 entity and averaged over all Source 1 entities. The supplied formula is:

F₀.₅ = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

The materials describe the score as precision-weighted and state that a false merge costs about twice as much as a miss. A true singleton (no matches in Source 2 or 3) scores 1.0 for an empty prediction and 0.0 if any match is predicted.

The problem statement says the public leaderboard uses one subset of test and the private leaderboard uses the remaining subset; submit the full test set in both cases. It says final decisions are based on the private leaderboard. See the wording conflict below: the event instructions also say shortlisting/evaluation uses both leaderboards.

## 9. Submission package and methodology

Every team must submit one ZIP package in addition to the live leaderboard upload. The described structure is:

- output/matching_results.tsv — final matches; same file uploaded to the Portal.
- output/candidate_pairs.tsv — final candidate set.
- code/business_entity_resolution/src/ — all source code.
- code/business_entity_resolution/README.md — exact end-to-end run instructions.
- code/business_entity_resolution/requirements.txt — pinned dependency versions or equivalent environment.
- Documentation_template.md — filled methodology document; do not rename it.

The code folder must be a self-contained, runnable copy that can reproduce both outputs from the training/test data. The methodology template must describe methodology, candidate generation/blocking, model architecture and feature engineering, and other relevant information. The problem statement says there is no page limit and asks for clarity and technical depth. A filled Markdown file is acceptable; PDF export is also acceptable. Top teams’ packages are reviewed before final rankings are confirmed.

The final model must have an MIT or Apache 2.0 license and be no larger than 8 billion parameters.

## 10. Fair-play rules

External data lookup to identify or resolve businesses is strictly prohibited. The listed examples include commercial entity-resolution APIs/services, government business-register lookups, geocoding APIs for address normalization, and internet-based data augmentation. The challenge says to use only the provided data, reviews submitted methods and code, and may immediately disqualify evidence of external data lookup.

The event instructions separately prohibit cheating, plagiarism, and unfair practices such as using multiple IDs; they state that these can result in immediate disqualification.

## 11. Challenge-provided tips, not selected team methods

The materials offer the following advice; these are not hard output requirements and have not been adopted as the team’s solution:

- Blocking sets the upper bound on recall because a record never considered cannot later be matched.
- False merges are costly; when uncertain, the video says not to merge.
- Account for region-specific name and address patterns.
- The written tips mention Jaccard, Levenshtein, and TF-IDF cosine as examples of string-similarity features.
- Correct singleton predictions matter.
- Validate output format before submission.

## 12. Organizer and event instructions

The event-instruction screenshots state:

- Challenge window: 25 September 2026, 12:00 AM IST through 27 September 2026, 11:59 PM IST.
- The problem statement and dataset are made available on day 1; teams can build and submit through day 3.
- Maximum of five submissions per team per day. The submit button is disabled after the daily cap.
- Maintain version history for every submission; shortlisting is based on submitted solutions. Final source code may be requested later.
- Participants use a desktop or laptop, not a mobile device.
- Simultaneous logins are prohibited; one desktop/laptop session per participant. The instructions say detected simultaneous logins may terminate the challenge session or cause errors.
- The instructions refer participants to a preparation blog for ML Challenge best practices/live demo and a Google Form for hackathon queries. Their actual URLs are not captured here; use the links in the official event portal.
- For technical problems, the instructions suggest clearing browser cache, using another browser/incognito, or switching internet connections. They direct participants to email support@unstop.com with a screenshot of the problem page and registered email ID. They say support will not make decisions for participants.

The event instructions say that after successful artifact submission, leaderboard score, and eligibility checks for each team member, the top 100 teams will be announced. They say the top 100 must provide methodology, candidate generation/blocking strategy, model architecture and feature engineering, and other relevant approach information. The instructions also mention a 1–2 page document explaining the ML approach, models used, experiments, and conclusion, plus commented source code for experiments, training, and inference.

## 13. Wording conflicts and open questions

Do not resolve these by assumption; retain them for official clarification if needed.

1. **Leaderboard basis:** The problem statement says final decisions are based on the private leaderboard. The event instructions say evaluation and shortlisting are based on performance across both public and private leaderboards. The materials do not say whether these refer to different decisions.
2. **Methodology length:** The problem statement says the filled Documentation_template.md has no page limit. The event instructions request a 1–2 page document. It is unclear whether the latter is a separate top-100 document or a limit on the same artifact.
3. **Country as a matching input:** The sign-up video says the challenge uses name and address, while the written record schema includes country and says it is an open set with France in test. The materials do not explicitly resolve whether country is a matching signal, metadata, or both. Do not drop the field or restrict its values.
4. **Eligibility:** Top-100 selection depends partly on team-member eligibility, but these screenshots do not define the eligibility criteria.
5. **Matching structure:** No global one-to-one constraint or ordering requirement for IDs inside a comma-separated list is stated. The explicit duplicate rules apply to repeated Source 1 rows and repeated IDs within a list.

## 14. Team engineering workflow — process only

No solution choices are made here. The intended engineering process is:

1. Convert the statement into testable contracts for parsing, labels, candidate output, predictions, score, fair play, licensing, and package structure.
2. Profile the actual files: counts, missingness, duplicates, source/country distributions, field formats, and match cardinality.
3. Audit label integrity and preserve the full zero/one/many match-set semantics.
4. Freeze a leakage-aware validation split and a faithful implementation of macro F₀.₅, including singleton scoring.
5. Measure candidate coverage and candidate volume separately from final matching quality.
6. Run controlled experiments only after the data and validation are trusted; log code/data split, changes, results, error categories, runtime, and memory.
7. Review false merges, missed links, blocking misses, singletons, multi-match entities, and country/source-pattern errors.
8. Stress-check generalization using training data only; do not use test labels or external lookups.
9. Reproduce both outputs from the packaged code, run the official validator, and document model license/size and dependencies before submission.

Exact candidate rules, features, model family, thresholds, and any global matching constraints remain undecided until evidence from the data and validation supports a choice.
