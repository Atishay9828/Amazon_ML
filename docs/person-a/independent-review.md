# Person A candidate generation: independent review and experiments

Date: 26 September 2026 (IST).
Scope: Person A retrieval code on branch `feat/data-candidates`, commits `78fc8f5` to `c4ac8e8`.
Status: C1–C4 runs are in progress. Section 9 lists them.

This document records the review findings, the measured experiments, and the recommended final retrieval configuration. The review started read-only. Later commits on this branch implement its recommendations in the repository. The experiment kernels ran on the private Kaggle account `pixel2005`; the experiment scripts are not part of the repository.

## 1. Summary

- The fixed `_rank` score was the largest loss in the final candidate set. A learned shortlist ranker removes most of that loss.
- On the broad-token configuration, the learned ranker at cap 16 gives 95.24% true-link recall. `_rank` at cap 64 gives 93.74%. The learned ranker uses one quarter of the pairs.
- The raw proposal ceiling of the broad-token configuration is 95.67%. Deeper staging or a larger final cap cannot pass this value.
- Three new proposal channels add recall above the ceiling:
  - Second hop (sibling expansion): net +137 links.
  - Hash-ordered character grams on the index side and the query side: net +61 links.
  - Reverse top-1 (each target retrieves its best Source 1 record): +308 links that no forward channel finds.
- The best measured union is 96.74% true-link recall at learned cap 32 plus reverse top-1. The C1 run measures the full combination with one ranker.
- A larger hash space (2^23) and exact-key channels under `_rank` give no net gain.

All recall values are candidate recall. They are not matcher F0.5 and not leaderboard F0.5. Section 8 explains the difference.

## 2. Evaluation protocol

All measurements use the same protocol, unless a row states a different sample.

- Queries: the seeded 10,000 training Source 1 IDs (`_select_source1(..., "dev")`, seed 20260926).
- True links: 34,770.
- Targets: the complete training Source 2 and Source 3 pools (10,320,219 records).
- Comparison: paired. Each variant reports links gained and links lost against the same reference file.
- Ranker training data: 20,000 training Source 1 IDs that are not in the dev or holdout selections (seed 7). The ranker never sees dev labels.
- Noise level: differences below approximately 0.5 points (about 170 links) are ties, unless the paired counts show a one-sided change.

## 3. Code review findings

### 3.1 Staging cut and ranking

- File: `code/business_entity_resolution/src/candidates.py`, `_run_chunk`, staging line (`candidates[:max(settings.stage_cap, ...)]`).
- The code sorts the complete proposal set by `_rank` and keeps `stage_cap` rows for each source. It writes the staged chunks after this cut.
- Result: the "staged recall" metric is recall after the first `_rank` cut. It is not raw proposal recall. The team corrected this interpretation on 26 September.
- `_rank` (`0.55·name + 0.45·address` plus small channel bonuses) has no information about country, house numbers, PIN codes, token overlap, or block size.

### 3.2 Character-gram index selection

- File: `candidates.py`, `_field_index` (gram selection) and `_best_field_hits` (query probe order).
- The original index keeps the rarest grams of each target. Rare grams are often grams that contain a typo. The Source 1 query does not contain those grams.
- Probe result (older baseline, 4,076 misses): 61.6% of missed name links were structurally unreachable by the name character channel. For found links, the value was 21.5%.
- Commit `b87563c` added `--gram-selection hash` for the index side only. The query side still uses the rarest-posting order. Section 5.3 shows that the query side matters.

### 3.3 Hash space

- Source 2 alone has 787,486 informative name tokens and 606,645 informative address tokens. The hash space has 1,048,576 buckets.
- Collisions occur. Experiment E1 shows that they do not reduce recall at this operating point.

### 3.4 Memory telemetry

- The original `_memory_state` reported the parent process only. Fork workers were not counted.
- Commit `b87563c` added container-level memory (`memory.current`, `memory.peak`). Observed container peak: 20.6–21.8 GB for all runs in this document. The Kaggle limit is approximately 30 GB.

### 3.5 Miss diagnostic

- File: `analyze_candidate_misses.py`. The original code used a per-source quota of 64 for cap-64 files. The true quota is 32 per source plus a shared remainder. The team corrected this.

### 3.6 Label leakage

- No label leakage was found in retrieval. `evaluate_retrieval`, `evaluate_staged`, and `aggregate_shard_reports` read labels only after generation.
- Dev and holdout selections are disjoint.
- Test IDF and document frequency use unlabelled test targets only. This is transductive and permitted.
- Risk: approximately 10 configurations were selected on the same 10,000 IDs. Run the final configuration once on the 50,000 holdout before the configuration is frozen.

### 3.7 Teammate report of 99.84%

- The report gives 6,975 of 6,986 links on 2,000 unknown IDs. The cap, pair count, target pool, and code are unknown.
- The team's own biased scratch corpus (true targets plus 50,000 distractors) gave 98.6%. A reduced target pool alone can produce a value of this size.
- Do not compare this value until the teammate supplies the IDs, the cap, the pair count, and confirmation that all 10.32 million targets were searched.

## 4. Diagnostic probe (older baseline, 4,076 misses)

Kernel: `pixel2005/amazon-ml-review-probe`. Reference: baseline cap-64 dev output.

| Measurement | Misses | Found links (sample of 4,000) |
| --- | ---: | ---: |
| Name channel structurally unreachable | 61.6% | 21.5% |
| Address channel unreachable (includes 503 blank target addresses) | 29.1% | 6.2% |
| Recovered by at least one exact key | 1,488 (36.5%) | – |
| Retrieved true sibling closer than Source 1 | 2,067 (50.7%) | – |
| Both raw name and address cosine below 0.3 | 50 (1.2%) | – |

- Other-country candidate slots: 33,689 of 636,560 (5.3%). True cross-country links in training: 0.
- Only 50 misses have no usable text similarity. Most misses are reachable in principle.

## 5. Experiment results

### 5.1 Round 1 (base: `word-pair-1`, K32/A16, 8/16 grams, cap 64)

Reference: the `_rank` control of E4. It reproduces the team's word-pair run exactly (31,356 links, 90.18%).

| Run | Change | Links | Gained | Lost | Net | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 | Hash space 2^20 to 2^23 | 31,339 | 171 | 188 | −17 | 90.13% |
| E2 | Hash-ordered grams, index and query | 31,730 | 594 | 220 | +374 | 91.26% |
| E3 | Exact-key channels with +0.10 rank bonus | 31,443 | 968 | 881 | +87 | 90.43% |
| E4 | Learned ranker, cap 64 | 31,831 | 481 | 6 | +475 | 91.55% |

- E3 found 968 new true links. The `_rank` bonus displaced 881 other true links. Channels with large blocks need a ranker that knows the block size.
- E4 at cap 16 (91.01%) is better than `_rank` at cap 64 (90.18%).

### 5.2 Proposal ceiling of the broad-token configuration (R4)

Configuration: HEAD `c4ac8e8`, K32/A16, 16/32 grams, pair DF 5,000, pair hits 2,000, two single-word probes at DF 512. Staging: 2,048 rows for each source. Order: `_rank`.

| `_rank` cutoff per source | 32 | 64 | 128 | 256 | 512 | 1,024 | 2,048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall | 93.74% | 94.99% | 95.54% | 95.65% | 95.67% | 95.67% | 95.67% |
| True links | 32,594 | 33,029 | 33,219 | 33,258 | 33,264 | 33,265 | 33,266 |

- Only 14 of 20,000 source groups reached 2,048 rows. 95.67% is effectively the raw proposal recall of the broad channels.
- 1,504 true links (4.3%) are not proposed at any depth.
- A stage cap of 256 for each source is sufficient.

### 5.3 Round 2 (base: broad-token configuration, stage cap 256, learned ranker v2)

| Run | Change | Staged-256 ceiling | Learned cap 16 | Learned cap 32 | Learned cap 64 | Net vs R0 learned cap 64 | Retrieval time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R0 | None (control reproduces 93.74%) | 95.65% | 95.24% | 95.44% | 95.51% | – | 2,172 s |
| R1 | `--gram-selection hash` (index only) | 95.83% | 95.23% | 95.50% | 95.57% | +100 / −80 = +20 | 2,522 s |
| R2 | Hash order on index and query | 95.80% | 95.38% | 95.60% | 95.69% | +108 / −47 = +61 | 2,480 s |
| R3 | Second hop (2 seeds, K16) | 96.16% | 95.62% | 95.84% | 95.91% | +182 / −45 = +137 | 2,526 s |

- R0 `_rank` control at cap 64 lost 629 of the R0 learned links and gained 13.
- R2 gains and R3 gains overlap little. 80 of 108 R2 gains are not in R3. The two changes should stack.
- Oracle macro F0.5, R0: 0.9770 (`_rank` cap 64) and 0.9835 (learned cap 16).

### 5.4 Reverse top-1 (RV)

Method: index all 2,206,821 training Source 1 records with concatenated `name | address` character 3-gram TF-IDF (document-frequency cap 1%). Each dev true target retrieves its exact top-k Source 1 records.

| Reverse depth | Reverse recall alone | Union with R3 learned cap 64 | New links | Extra test pairs (upper bound) |
| --- | ---: | ---: | ---: | ---: |
| Top-1 | 88.55% | 96.79% | +308 | 10.0 M |
| Top-2 | 90.33% | 96.91% | +348 | 19.9 M |
| Top-3 | 91.20% | 96.97% | +370 | 29.9 M |
| Top-5 | 92.09% | 97.06% | +401 | 49.8 M |

- R3 learned cap 32 plus reverse top-1: **96.74%**, with 313 new links.
- Each target has at most one true Source 1 record. The reverse direction is not affected by the crowding that fills a Source 1 shortlist with chain branches and neighbouring businesses.
- Index build time for 2.2 million Source 1 records: 86 s.

### 5.5 Exact forward top-k on concatenated 3-grams (FX)

Method: index each target source with concatenated `name | address` character 3-gram TF-IDF (document-frequency cap 1%). Each dev Source 1 record retrieves its exact top-64 targets from each source.

- FX alone: 85.46% at top-4, 91.27% at top-16, and 93.80% at top-64 for each source. This is lower than the tuned approximate channels.
- Union with R3 learned cap 32 plus reverse top-1 (96.74%):

| FX depth for each source | New links | Union recall | Extra pairs for each Source 1 record |
| --- | ---: | ---: | ---: |
| 4 | +67 | 96.94% | +8 |
| 8 | +114 | 97.07% | +16 |
| 16 | +149 | 97.17% | +32 |
| 32 | +209 | 97.35% | +64 |
| 64 | +284 | 97.56% | +128 |

- Runtime: index build 300 s for each source. Query time is approximately 36 ms for each query and source on one core. A full test run needs approximately 17 hours for each source on one core.
- Decision: FX is optional. Use it only with a lower document-frequency cap and 4 workers, and only if a measured shard fits the time budget. The reverse channel gives more recall for each unit of cost.

### 5.6 Combined configuration C1

Configuration: broad-token channels, hash order on index and query, second hop (2 seeds, K16), reverse top-1 (document-frequency cap **0.002**), stage cap 256, learned ranker v2.

| Measurement | Value |
| --- | ---: |
| Staged-256 ceiling | 96.33% |
| Learned cap 16 / 32 / 64 | 95.78% / 96.04% / 96.16% |
| `_rank` control cap 64 | 94.30% |
| Reverse pass over 10,320,219 train targets (4 workers) | 1,725 s |

- C1 is 0.7 points below the union estimate (96.74%). The union estimate used the reverse index with document-frequency cap 0.01. At cap 0.002 the reverse channel stages only about 60 new links instead of about 310.
- Decision: use document-frequency cap 0.01 for the reverse index. The estimated reverse pass time at 0.01 is approximately 2–2.5 hours for each split.

### 5.7 Residual misses and new exact keys

After all channels (R3 cap 64, reverse top-5, FX top-64), 812 dev links (2.33%) remain. The groups:

| Group | Misses | Example |
| --- | ---: | --- |
| Target name in an Indian script | 366 | "International Management Private Limited" to "इंटरनेशनल मैनेजमेंट प्राइवेट लिमिटेड" |
| Blank target address with a generic name | 264 | "International Hospitality Limited" to "International Hospitality" |
| Shared words with heavy typos (digit-letter swaps) | 170 | "K6D Associates", "C0mmittee" |
| Random or website name | 83 | "Sector45bridge.Com" |

Four exact keys address these groups (`src/match_keys.py`): a coarse consonant-skeleton name key that makes English names and their Indic-script transliterations collide, the same key after OCR digit repair, a compact name key that keeps digits, and street numbers with the first name code. Key probe over all 10.3 million targets, against R3 cap 32 plus reverse top-1 (1,132 misses):

| Keys (block cap) | New links | Extra pairs for each Source 1 record, before ranking |
| --- | ---: | ---: |
| Phonetic (200) | +269 | 24 |
| OCR phonetic (200) | +258 | 25 |
| Compact with digits (200) | +160 | 18 |
| Numbers and first code (200) | +179 | 10 |
| All four (50) | +266 | 11 |
| All four (200) | +469 | 44 |
| All four (2,000) | +572 | 144 |

Single-field reverse search (RF):

| Channel | New links | Extra test pairs |
| --- | ---: | ---: |
| Name-only reverse for blank-address targets, top-10 | +96 | about 3.3 M |
| Name-only reverse for blank-address targets, top-20 | +126 | about 6.7 M |
| Address-only reverse, top-1 | +4 | about 10 M (rejected) |

C5 (C1 plus the four keys at block cap 200 and key-equality ranker features) is running. P1 runs the committed repository pipeline end to end with `configs/final.json`.

## 6. Recommended final configuration

The C1 run measures this configuration as one unit. Confirm the C1 result before the configuration is frozen.

1. Use the broad-token proposal channels of HEAD `c4ac8e8`.
2. Set `--gram-selection hash` and apply the same hash order to the query probe in `_best_field_hits`.
3. Add the second-hop channel with 2 seeds and K16.
4. Add the reverse top-1 channel. Build it on the **test** Source 1 records for the test run. Keep every reverse pair through the staging cut.
5. Set `--stage-cap 256`.
6. Replace `_rank` for the final selection with the learned ranker. Train it on training IDs outside dev and holdout. Use at least 50,000 IDs for the final model.
7. Use a final cap of 16 or 32 for each Source 1 record. Cap 32 is the safer choice for recall. Cap 16 reduces the scoring load of Person B by half again.
8. Run the final configuration once on the 50,000 holdout. Do not tune after this run.

Estimated test volume: 1,732,544 × 32 ≈ 55 M forward pairs, plus at most 10 M reverse pairs. The current cap-64 file has 110 M pairs.

## 7. Implementation notes for the repository owner

A copy of the reference code is in `review\reference-code\`, next to this document. The code is experiment code. It is not production code.

| Item | Scratchpad file | Location in file |
| --- | --- | --- |
| Learned ranker v2 features and training | `launcher2.py` | Block after `import numpy as np, pandas as pd` |
| Second hop | `cand_hop.py` | Block after `if settings.hop_seeds:` in `_run_chunk` |
| Hash order on the query side | `cand_hashq.py` | `_best_field_hits`, `if settings.gram_selection == 'hash'` |
| Reverse top-1 and injection | `launcher_c1.py`, `cand_combo.py` | "Reverse top-1" block; `_load_extra` and `reverse_top1` |
| Proposal ceiling curve | `launcher2.py` | `if RUN == "r4":` block |

### 7.1 Ranker v2 features

- Scores: `name_cosine`, `address_cosine`, product, maximum, minimum, `base_rank` (the `_rank` value).
- Positions inside the source: `_rank` position, name-cosine position, address-cosine position, number of rows for the source.
- Text overlap: name token Jaccard, address token Jaccard, number-token Jaccard (ordinal suffixes removed), number agreement flag, number conflict flag.
- Keys: name-signature equality, compact-name equality (8 or more characters).
- Other: blank target address, same country string, target from Source 3, number of channels, one flag for each channel in `CHANNEL_ORDER`.
- Model: `sklearn.ensemble.HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0)`.
- Selection: sort by model probability. Keep `cap // 2` rows for each source, then fill the remainder by probability. This is the same quota logic as `_choose_rows`.

### 7.2 Second hop

For each Source 1 record, compute the cosine scores of the first-hop proposals. Select up to 2 seeds with name cosine ≥ 0.8 and address cosine ≥ 0.6 (or name cosine ≥ 0.95). Use the TF-IDF row of each seed as a query to `_best_field_hits` (name K16, address K8). Add the results with the channel `second_hop`. No labels are used.

### 7.3 Reverse top-1 for the test run

1. Build a concatenated `normalize_name | normalize_address` character 3-gram TF-IDF index on the test Source 1 records. Remove grams whose document frequency is above 0.2% to 1% of the records.
2. Stream all 9,969,589 test targets in batches of 256. Use 4 fork workers that share the transposed index.
3. Keep the top-1 Source 1 record for each target.
4. Write the pairs as `source1_entity_id<TAB>candidate_entity_id`.
5. Pass the file with `--extra-pairs`. The pairs get the channel `reverse_top1` and pass the staging cut.

The C1 run reports the full reverse-pass runtime for 10.3 million training targets. Use this value to plan the test run.

## 8. Metric definitions

- **Candidate recall:** the fraction of true links that reach Person B. This document reports this metric.
- **Oracle macro F0.5:** the score of a perfect matcher on the candidate set. The metric gives every singleton 1.0. A real matcher must reject all candidates of a singleton to get that value. Larger candidate sets make this more difficult. Oracle gains therefore overstate real gains.
- **Matcher F0.5 and leaderboard F0.5:** only the Person B model on the final candidate set produces these values. No retrieval result predicts them.

Recommendation for Person B: for each Source 1 record, sort the candidates by probability. Compute the expected F0.5 for each prefix size, including the empty prefix. Output the prefix with the highest expected value. The expected value of the empty prefix is the probability of no match. This rule handles singletons directly (Ye et al., ICML 2012; Waegeman et al., JMLR 2014).

## 9. Runs in progress

| Kernel | Configuration | Question |
| --- | --- | --- |
| `amazon-ml-review-c1` | Broad + hash (index and query) + second hop (2, K16) + reverse top-1 + ranker | Recall of the recommended configuration |
| `amazon-ml-review-c2` | C1 + `--name-keys` | Do name keys add recall when a learned ranker controls displacement? |
| `amazon-ml-review-c3` | C1 with second hop 4 seeds, K32 | Does a larger sibling expansion add recall? |
| `amazon-ml-review-c4` | C1 without hash grams | Is the hash runtime cost (about 14%) worth the gain inside the combination? |

The FX run is complete. Section 5.5 gives its result. Add the C1–C4 results to this document when the runs complete.

## 10. Research references

- Paulsen et al., "Sparkly: A Simple yet Surprisingly Strong TF/IDF Blocker for Entity Matching", VLDB 2023. Top-k BM25 over 3-grams. Top-k is more predictable than thresholds. Probing from the larger table gives higher recall. <https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf>
- Papadakis et al., "Supervised Meta-blocking", VLDB 2014, and "Generalized Supervised Meta-blocking". Candidate pruning as classification. Block-statistics features such as ARCS (sum of inverse block sizes). <http://www.vldb.org/pvldb/vol7/p1929-papadakis.pdf>, <https://arxiv.org/pdf/2204.08801>
- Papadakis et al., "A Survey of Blocking and Filtering Techniques for Entity Resolution". <https://arxiv.org/pdf/1905.06167>
- "Towards Universal Dense Blocking for Entity Resolution" (UniBlocker). Dense blockers are comparable to and complementary with sparse blockers. <https://arxiv.org/pdf/2404.14831>
- Ye, Chai, Lee, Chieu, "Optimizing F-Measures: A Tale of Two Approaches", ICML 2012. <https://icml.cc/2012/papers/175.pdf>
- Waegeman et al., "On the Bayes-Optimality of F-Measure Maximizers", JMLR 2014. <https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf>

The reviewer did not open public repositories of other challenge participants. The fair-play rules of the challenge forbid plagiarism.

## 11. Next steps for the team

1. Record the C1–C4 and FX results in this document.
2. Implement the items in Section 7 in the repository. Codex can do this work in parallel with the running experiments.
3. Freeze the configuration. Run the 50,000 holdout once.
4. Generate the full training candidates with the frozen configuration for Person B. The matcher must train on the same candidate distribution that it scores on the test set.
5. Generate the full test candidates. Measure the runtime of one shard first.
6. Person C writes the wide `candidate_pairs.tsv` with one row for every test Source 1 record, including records without candidates.

## Appendix A. Artifacts

- Experiment code: `scratchpad\exp\` (round 1) and `scratchpad\round2\` (round 2).
- Downloaded outputs: `scratchpad\exp-out\` and `scratchpad\r2-out\`.
- Kaggle account: `pixel2005`. Private datasets: `amazon-ml-2026-review-input` (the seven official TSV files) and `amazon-ml-2026-review-code`. Private kernels: `amazon-ml-review-*`.
- A copy of the reference code is in `review\reference-code\`. The scratchpad folder is session-specific.
