# AJ experiment: official-data audit

This report summarizes the locally extracted challenge files. It contains no row-level challenge records or predictions. The sampled similarity figures are diagnostics, not a measured candidate-retrieval score.

| Split | Source 1 | Source 2 | Source 3 |
| --- | ---: | ---: | ---: |
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

The seven TSV files total approximately 2.52 GB. Every source file has `entity_id`, `business_name`, `business_address`, and `country`; the training ground truth has `source1_entity_id` and `matched_entity_ids`. IDs are unique within sources and do not overlap across the train and test splits. Every training Source 1 ID has exactly one ground-truth row. Every listed match points to a real training Source 2 or Source 3 ID, and no target ID appears in more than one ground-truth list.

Training has **7,638,365 labeled links**: 3,693,619 to Source 2 and 3,944,746 to Source 3. There are 123,247 singleton Source 1 rows (5.59%), and the maximum observed match count is 11. Some 1,776,047 Source 1 rows (80.48%) link to both target sources. Approximately 1.34 million rows in each training target source are not listed as matches. All 7,638,365 labeled links have identical country labels on both sides.

Train contains US and India. Test also contains France, with **259,452 French Source 1 rows** (about 15% of test Source 1). Source 1 names and addresses are present. Target addresses are blank in roughly 3.3% of the training target records and 2.7% of the test target records.

On a deterministic sample of **37,552 positive pairs**, exact normalized name matches cover 21.93% and exact normalized nonblank address matches cover 8.23%; their union covers 28.91%. At least one normalized core-name token overlaps in 85.60% of sampled pairs, and an address token overlaps in 95.55%. Indian name overlap is materially weaker than US name overlap. A sampled union of rare name/address token keys, exact keys, and numeric-address keys reaches 99.59%, but the token frequencies in that calculation came from the sampled positives. **That 99.59% is not the full-index candidate recall.** The notebook measures the latter on held-out training entities after its real posting-list limits and per-query caps are applied.

Implications for this experiment: process one country and target source at a time; keep the country set dynamic; use both name and address retrieval channels; preserve multiple links per Source 1; score singletons in the macro metric; stream full-test inference; and inspect actual candidate recall before any portal submission. France has no labeled validation score.
