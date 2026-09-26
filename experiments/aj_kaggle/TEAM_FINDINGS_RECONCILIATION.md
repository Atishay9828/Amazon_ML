# Team findings: evidence and interpretation

Date: 2026-09-26

This note incorporates the findings reported in AJ's team screenshots. The counts below are **reported team evidence**, not a new full-dataset audit performed for this note. They are reconciled with the earlier [DATA_AUDIT.md](DATA_AUDIT.md), the current baseline source, and the reported run results in the [experiment README](README.md). No challenge records or predictions are included here.

## 1. Reported dataset counts

| Split | Source | Records | Missing addresses reported | Missing-address denominator |
| --- | --- | ---: | ---: | --- |
| Train | Source 1 | 2,206,821 | None reported | Train Source 1 records |
| Train | Source 2 | 5,034,616 | 168,967 | Train Source 2 records: 3.3561% |
| Train | Source 3 | 5,285,603 | 175,916 | Train Source 3 records: 3.3282% |
| Test | Source 1 | 1,732,544 | None reported | Test Source 1 records |
| Test | Source 2 | 4,887,273 | 129,408 | Test Source 2 records: 2.6479% |
| Test | Source 3 | 5,082,316 | 136,098 | Test Source 3 records: 2.6779% |

The screenshots report no missing business names or country labels. These presence counts do not establish that names or country labels are accurate or sufficient for matching. The screenshot's missing-address definition should not silently be expanded to include punctuation-only addresses or other derived empty values; those can be counted separately by preprocessing diagnostics.

The row counts agree with the earlier audit. Summing all three source files gives **12,527,040 training records** and **11,702,133 test records**. These totals count source records, not distinct real-world businesses.

## 2. Reported training-label integrity

| Finding | Reported evidence | Scope of the conclusion |
| --- | ---: | --- |
| Labeled links | 7,638,365 | Links in training ground truth |
| Linked IDs | All listed IDs exist | Referential integrity of the checked training labels |
| Cross-country positive links | 0 observed | Observed training links only |
| Source 1 singletons | 123,247 | Training Source 1 entities with no listed match |
| Largest true-match list | 11 | Maximum observed combined Source 2/Source 3 truth count for one training Source 1 entity |

The singleton denominator is **2,206,821 training Source 1 entities**:

`123,247 / 2,206,821 × 100 = 5.58482088%`.

That is approximately 5.585%, or **5.58% rounded to two decimal places**, matching the screenshot. The earlier local audit's 5.59% has been corrected; the exact count and denominator should be retained in experiment records.

Zero observed cross-country training links supports evaluating same-country candidate generation. It does not prove that country labels are semantically error-free or that the unlabeled test set has the same property. France has no labeled training examples, so this evidence cannot establish French test recall or accuracy.

## 3. Country percentages require a named denominator

Every reported country share must state both its split and its counting unit:

- **Source 1 share:** country Source 1 rows divided by all Source 1 rows in that split.
- **All-source record share:** country rows across Sources 1, 2, and 3 divided by all rows across those three sources in that split.

For France, the supplied Source 1 count gives:

`259,452 / 1,732,544 × 100 = 14.97520409%`.

Therefore, France represents **14.98% of test Source 1 records**. This is not an all-source percentage. A share across all test records would require the French Source 2 and Source 3 counts as well; the Source 1 numerator cannot be divided by an all-source denominator and presented as the same statistic.

## 4. Eleven true matches does not make a candidate cap safe

The claim that a cap of 500 cannot truncate true matches because the largest observed truth list has 11 is **not established**. The two quantities measure different things:

- The truth-list maximum counts actual matches for a Source 1 entity.
- The candidate cap limits all retained candidates, including distractors, after ranking and any earlier filtering.

A true match can rank below 500 distractors and be removed even when it is the entity's only true match. Earlier token-frequency or key-selection filters can remove it before candidate ranking begins. A safe-cap claim would require evidence that every relevant true match survives those earlier gates and ranks within the chosen cap; the maximum truth cardinality alone provides no such evidence.

### Keep the configurations separate

| Configuration | Candidate cap | What is established |
| --- | --- | --- |
| AJ's current baseline in [one_cell_solution.py](one_cell_solution.py) | **40 per target source**, allowing up to **80 combined** for one Source 1 entity | Source 2 and Source 3 are capped separately |
| Candidate configuration shown in the team screenshot | **500 reported** | Another team configuration; its scope and other retrieval settings must be recorded before comparison |

The screenshot's 500 must not be used to describe AJ's baseline. If the screenshot does not establish whether 500 is per source or combined, that scope remains unspecified.

The experiment README already records **0.840123886140238 true-link candidate recall** from AJ's supplied run log. This is a reported measurement for that run, not a new measurement in this note and not a result for the screenshot's 500-cap configuration. It shows that some labeled validation links did not reach the matcher; it does not yet attribute those losses to the final cap versus earlier gates.

## 5. Present names do not guarantee recovery of missing addresses

The claim that names recover missing-address cases without any penalty is **not established** by the absence of missing names. A populated name may still be altered, ambiguous, or removed by a retrieval gate. Candidate ranking can also place its true match below the cap.

The necessary comparison is a **blank-target-address recall slice** against a nonblank-target-address slice, using the same query selection, full target-country index, preprocessing, key limits, ranking, and cap. Its denominator must include every labeled true link in the slice, including links that produced no candidate.

Required reporting for that comparison:

1. The exact blank-address definition and number of labeled links in each slice.
2. Candidate-link recall by target source and country, with zero-candidate cases retained.
3. Losses attributable to key generation, frequency gates, key-count limits, and final candidate ranking/capping.
4. Candidate counts and cap saturation alongside recall, so a wider shortlist is not mistaken for a free improvement.

No blank-address recall result is claimed here.

## 6. Scope of the phrase “zero corruption”

Use this phrase only with the checks named. The reported checks concern items such as valid IDs, expected source prefixes, referenced-ID existence, and tab-separated parsing/field structure. They do not establish semantic cleanliness of business names, addresses, or country labels.

The challenge explicitly contains noisy and inconsistent representations. Structural validity, field presence, normalization correctness, candidate recall, and final matching accuracy are separate measurements.

## 7. Where the checks and measurements belong

| Question | Existing artifact or required measurement | Status in this note |
| --- | --- | --- |
| Do the reported full-file counts align with earlier context? | [DATA_AUDIT.md](DATA_AUDIT.md) | Counts align; singleton percentage arithmetic clarified above |
| Does preprocessing preserve raw values, Unicode marks, address numbers, and suffix boundaries? | [preprocessing.py](preprocessing.py) and its [synthetic tests](tests/test_preprocessing.py) | Implemented and tested separately; synthetic correctness is not retrieval recall |
| Where are true links lost in retrieval? | [diagnostics.py](diagnostics.py): development-only sampled Source 1 queries against full target-country indexes | Gate and ranking attribution must come from an actual completed diagnostic run |
| Does cap 500 retain more true links than the baseline? | Controlled cap comparison with all other retrieval settings fixed | No new result claimed here |
| Do blank-address cases lose recall? | The explicitly defined slice comparison above | No new result claimed here |
| What has the existing baseline reported? | [Experiment README](README.md), with its supplied-log provenance | Keep those run-specific measurements separate from team screenshot claims |

The earlier audit's **99.59% sampled key-union coverage** remains an exploratory statistic based on token frequencies from sampled positive targets. As [DATA_AUDIT.md](DATA_AUDIT.md) already states, it is not full-index candidate recall after production frequency gates and caps. Neither that statistic nor the maximum truth-list size guarantees that a candidate configuration preserves every true link.

## 8. Person A's newly reported 2,000-query result

AJ supplied a further team screenshot reporting this experiment:

| Item | Reported value |
| --- | ---: |
| Training Source 1 queries | 2,000 |
| Labeled true links across Sources 2 and 3 | 6,986 |
| True links found by the candidate engine | 6,975 |
| Missed true links | 11 |
| True-link candidate recall | **6,975 / 6,986 = 99.842542%** |
| Search population | All Source 2 and Source 3 records, described as 10+ million |

The arithmetic agrees with the screenshot's 99.84%. This is **reported candidate recall on that sample**, not model precision, macro F0.5, full-training recall, or test-set performance. The screenshot does not identify the sample membership, code revision, final candidate cap, total candidate count, or whether the count was taken after every filtering stage.

### Comparison with the available Person A branch

The read-only review used commit [`b78862b165db3ab98bad5b7cca090e976a766847`](https://github.com/Atishay9828/Amazon_ML/tree/b78862b165db3ab98bad5b7cca090e976a766847), from `feat/data-candidates`. Its `docs/data_audit.md` and `docs/person-a/todo.md` document a **different, 10,000-query full-target development run**:

| Final cap, combined across both target sources | Candidate pairs | Documented true-link recall |
| --- | ---: | ---: |
| 16 | 160,000 | 83.1435% |
| 32 | 320,000 | 85.7981% |
| 64 | 636,560 | 88.2773% |
| 128, diagnostic | 1,114,940 | 89.2436% |

These are values recorded in Person A's documents; they were not independently rerun here. The inspected revision does not contain the 6,975/6,986 result. The new screenshot may describe a later or differently configured run, so the two sets of numbers must remain separate until its run artifacts are available.

That published engine uses exact keys, rare tokens, and character-gram retrieval with transliteration. Its default intermediate shortlist is 64 per source. Raising only the final combined cap to 500 cannot restore candidates lost before that stage. Its documented full-source peak process RAM is 19.946 GiB; it is not suitable for an additional full-target run under this local diagnostic's 512 MB DuckDB budget. Candidate retrieval is CPU work in that implementation; AJ's model training remains on GPUs.

### Required handoff to B and C

Person A's work is ready for integration when the handoff contains:

1. The exact code commit and configuration for the reported run, including all intermediate limits and whether each cap is per source or combined.
2. Private query membership and the sampling rule, plus input-file identities. Keep the original validation split excluded when comparing with AJ's development diagnostic.
3. The **final candidate set actually passed to matching**, with counts before and after each filter, and a report recomputed from that file against the complete selected-query truth denominator.
4. Recall by country, target source, and missing-address slice; the 11 misses with private diagnostic explanations; total candidates and per-query distribution; measured time and peak RAM.
5. Full-run completion records, output identities, and coverage/uniqueness/existing-ID checks for every required Source 1 entity, including empty candidate lists and France in test.

B can prepare matching and evaluation against the agreed interface while A produces the candidates. C can maintain integration, experiment records, and output checks in parallel. The message's proposed sequence does not require B and C to remain idle, and merely completing a large run does not establish that its final handoff is valid.

The controlled comparison in [diagnostics.py](diagnostics.py) uses the same frozen development queries for all three AJ policies and scans all target records in each source/country. It does not reproduce Person A's different engine or establish the new reported 99.84% claim.
