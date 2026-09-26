# Person A data audit — 26 September 2026 IST

The official TSVs were read from `../student_resource/dataset`, outside Git. Command from repository root:

```powershell
python code/business_entity_resolution/src/data.py --data-root '..\student_resource\dataset' > 'D:\temp\amazon-ml-person-a-audit.json'
```

`data.py` checks the exact tab-separated headers, row widths, source ID prefixes, and duplicate IDs for every source row. It also checks truth ID prefixes/uniqueness, duplicate matches, label coverage, target existence, and cross-country true links. The run exited 0. The complete machine-readable audit with sizes and SHA-256 hashes is at the private local path above; it is not committed because the repository is public.

| File | Rows | Country counts | Blank addresses | Mean name / address length |
| --- | ---: | --- | ---: | ---: |
| train Source 1 | 2,206,821 | US 1,323,633; India 883,188 | 0 | 24.034 / 52.066 |
| train Source 2 | 5,034,616 | US 3,016,817; India 2,017,799 | 168,967 | 25.104 / 46.226 |
| train Source 3 | 5,285,603 | US 3,170,056; India 2,115,547 | 175,916 | 25.202 / 46.714 |
| test Source 1 | 1,732,544 | US 663,106; India 809,986; France 259,452 | 0 | 23.836 / 57.213 |
| test Source 2 | 4,887,273 | US 1,871,330; India 2,312,565; France 703,378 | 129,408 | 25.704 / 50.413 |
| test Source 3 | 5,082,316 | US 1,945,701; India 2,405,000; France 731,615 | 136,098 | 25.655 / 48.740 |

No source file had a blank name or country, malformed row, duplicate ID, or incorrect source prefix. Source 2/3 blank addresses require name-only retrieval; no blank-blank exact block is allowed.

## Ground truth

- 2,206,821 labeled Source 1 rows, covering every training Source 1 ID exactly once.
- 7,638,365 positive links; no nonexistent target references and no cross-country positive links were found. This does **not** justify hard country filtering, especially because France appears only in test.
- 123,247 true singletons (5.58% of Source 1, rounded from 5.5848%). Maximum true matches per Source 1 is 11.
- Match-count distribution (0 through 11): 123,247; 119,157; 375,212; 530,841; 484,115; 321,957; 164,868; 63,968; 18,680; 4,205; 534; 37.

## Input fingerprints

| File | SHA-256 |
| --- | --- |
| train Source 1 | `591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973` |
| train Source 2 | `6336c1a055eec79cf8a6d99fdc8d32a2e4d9dc2662e00963cb35d66b89ed09ed` |
| train Source 3 | `67da22f5151898ff3006febd836c1a159e97ae95efa7257a5aff4fda685e58e9` |
| train truth | `70bc1d8a16c667e0155c2105d0ab2ebe41d7e7a85d8a529e3ca81c6c3a5af037` |
| test Source 1 | `3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5` |
| test Source 2 | `79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd` |
| test Source 3 | `850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035` |

## Retrieval implications and limits

The earlier seeded 5,000-Source-1 exploratory sample observed 28.31% true-link coverage by exact normalized name or address. Its method was not yet checked into source, so treat that figure as preliminary; it is not full-corpus blocking recall. The full audit above establishes integrity and scale, not any candidate or matching score. No test labels exist, so France accuracy cannot be measured locally.

### Biased retrieval diagnostic, not a full-corpus estimate

To exercise real noisy positives before a full run, Codex selected the **first 1,000 training Source 1 records**, kept their complete truth sets (3,517 positive links), and for each target source kept its first 50,000 rows **plus every positive target ID** for those 1,000 Source 1 records. The resulting private scratch corpus had 51,695 Source 2 and 51,792 Source 3 rows and all selected positive IDs. Because almost all full-corpus distractors are absent, every recall/oracle figure below is optimistic and must not choose the final setting.

| Retrieval variant, 64 or 32 total candidates/S1 | True-link recall | Complete nonempty true set | Oracle macro F0.5 | Pairs |
| --- | ---: | ---: | ---: | ---: |
| Name tokens + char grams, cap 64 | 0.9369 | 0.8436 | 0.9815 | 62,989 |
| Add rare address tokens, cap 64 | 0.9861 | 0.9556 | 0.9968 | 63,911 |
| Add rare address tokens, cap 32 | 0.9821 | 0.9429 | 0.9950 | 32,000 |
| Add generic script transliteration and vote-ranked gram probes, cap 32 | 0.9861 | 0.9556 | 0.9968 | 32,000 |

The biggest observed misses before address tokens had no shared normalized name token, often because one source used a different script; 155 of 222 missed links fell in that category, 62 had a shared name token, and 5 were present before final cap. Rare address tokens recovered most of them. The most recent variant uses [AnyAscii](https://github.com/anyascii/anyascii) for generic Unicode transliteration; it performs no business lookup. These experiments are **smoke tests**, not a leakage-aware Person B development split or leaderboard score.

### Local million-target memory benchmark

Codex ran `candidates.py --split train --limit-source1 100 --limit-targets 1000000 --query-chunk 100 --matrix-chunk 100000 --cap 32` against the official files on a 16 GB Windows PC (26 Sep IST). Source 2 indexing took 346.30 s and Source 3 indexing 367.62 s; each had 1,000,000 targets. Process peak working set was 3.20 GiB and the complete run emitted 3,200 pairs. This establishes bounded memory at one million targets; it does not establish full-source RAM, runtime, or candidate recall. The 100 Source 1 records were a file prefix, so this benchmark is only for resources.

The private Kaggle CPU smoke used 1,000 Source 1 records and the first 1,000,000 targets per source with two fork workers. It indexed Source 2 in 114.68 s and Source 3 in 117.44 s, with reported process peak RSS of 2.483 GiB; output contained 32,000 pairs. The five bundled tests passed in Kaggle's Python 3.12.13 environment, and `validate_candidate_long.py` subsequently confirmed all 32,000 pair rows, sorted uniqueness, score/channel contract, and target ID existence against the complete official train sources. This still provides **no full-corpus recall estimate**, because it omits most target records.

The seeded 10,000-row Kaggle development job built the **complete** train Source 2 index of 5,034,616 targets in 840.78 s and the Source 3 index of 5,285,603 targets in 895.23 s. The reported process peak RSS was 19.946 GiB, below the private CPU notebook's available memory. Four fork workers handled retrieval. Its cap-32 output contained 320,000 pairs and covered 85.7981% of true links, with 0.939228 oracle macro F0.5. This is a retrieval ceiling on the seeded development sample, **not** a trained matcher or leaderboard score.

Recapping the same staged retrieval at cap 16 yielded 160,000 pairs, 83.1435% true-link recall, and 0.923140 oracle macro F0.5. Cap 64 yielded 636,560 pairs, 88.2773% recall, and 0.953626 oracle macro F0.5. A diagnostic cap 128 yielded 1,114,940 pairs, 89.2436% recall, and 0.957544 oracle macro F0.5. The small gain from cap 64 to 128 shows that raising only the **final** cap has limited value when each source was already cut to 64 staged rows. It does not distinguish never-proposed links from links removed by the first `_rank` cut. The 50,000-row holdout remains untouched while development diagnostics continue.

Miss analysis of the diagnostic cap-128 table found 3,740 missed true links: 2,455 share at least one normalized name token **and** address token with their Source 1 record, 485 share a name token only, and 800 share an address token only. In 2,453 misses, that target source has fewer than its staged 64 candidates for the Source 1 record, indicating a proposal gap. In 1,287 misses, the target source is saturated at 64, so the staged per-source limit or ranking may be responsible. These categories are simple token overlap diagnostics, not proof that all such links can be recovered economically. The report with record examples stays outside Git at `D:\temp\amazon-ml-dev-misses.json`.

### Independent 50,000-row holdout check

The private Kaggle `amazon-ml-person-a-holdout-50k` run completed on the disjoint seeded 50,000 Source 1 selection, with the full 10,320,219 training targets and the same name K=32, address K=16, final cap 64 configuration. Its report recorded 172,581 true links, 3,182,916 candidate pairs, 88.1841% true-link recall, 72.45% complete-set coverage across all Source 1 records, and 0.951459 oracle macro F0.5. Mean candidates were 63.658 and p95 was 64. This is a retrieval check; it is not Person B's connected-component holdout or an actual matching score. The report was read once after development tuning, and no settings were chosen from its labels.

### Larger-neighbor development comparison

The private Kaggle `amazon-ml-person-a-dev-k64-a32` run used the **same** seeded 10,000 Source 1 IDs and complete 10,320,219 training targets as the baseline, with 64 name neighbors and 32 address neighbors per target source. At final cap 64 it retrieved 31,047/34,770 true links (89.2925%), emitted 640,000 pairs, and reached 0.957808 oracle macro F0.5. Peak reported process RSS was 20.113 GiB. The baseline at the same cap retrieved 30,694/34,770 links (88.2773%) and reached 0.953626 oracle macro F0.5. This gain is measurable but still leaves 3,723 true links unseen on development.

The subsequent cross-field token experiment `amazon-ml-person-a-dev-cross-token` kept K32/A16 and cap 64 on those same IDs/targets. It retrieved 30,965/34,770 true links (89.0567%), emitted 636,563 pairs, and reached 0.956852 oracle macro F0.5, with 19.934 GiB peak process RSS. A fresh missed-link audit found 3,805 true links absent from its final candidate table. This is an improvement over baseline but below the larger-neighbor experiment.

The combined `amazon-ml-person-a-dev-cross-token-k64-a32` run used cross-field tokens with K64/A32 at the same final cap 64. It retrieved 31,232/34,770 links (89.8246%), emitted 640,000 pairs, and reached 0.959808 oracle macro F0.5. Peak process RSS was 20.826 GiB; 3,538 true links remained unretrieved.

The same-field `name_pair`/`address_pair` run `amazon-ml-person-a-dev-word-pair` used K32/A16 and cap 64 on the same IDs/targets. It retrieved 31,356/34,770 links (90.1812%), emitted 636,937 pairs, and reached 0.962473 oracle macro F0.5, with 20.827 GiB peak process RSS. It still misses 3,414 true links. A separate all-labeled-training evaluation with word-pair retrieval and K64/A32 completed at 90.6929% recall; see the full-training section below. The different neighbor budgets prevent a strict sample-to-full comparison.

The wider-character-gram run `amazon-ml-person-a-dev-word-pair-grams16` kept the same word-pair channels, K32/A16 and final cap 64, but indexed 16 rare character grams per target field and probed 32 per query field (versus 8/16). It retrieved 32,028/34,770 true links (92.1139%), emitted 636,832 pairs, and reached 0.970748 oracle macro F0.5. Indexing peak process RSS was 20.467 GiB; the Kaggle complete event was at about 2,007 seconds. The later broad-token run improved the development result to 93.7417%. The full-training shards use the earlier 8/16-gram word-pair configuration, so their measurements do not establish the newer method's full-data performance.

The stage/final comparison for the earlier word-pair K32/A16 development run explains part of the remaining gap. Its 10 per-source staged chunks contained 1,126,028 pairs and 31,723/34,770 true links (91.2367% stage recall); the final cap-64 file retained 31,356 true links (90.1812%). Thus 3,047 true links were absent **after the first `_rank` cut to 64 per source**, and 367 more were lost at final selection. The absent 3,047 may mix never-proposed links with proposed links ranked below 64; the current artifacts cannot separate those causes. Raising only the final cap cannot reach 98% on that run. The staged files and comparison output remain outside Git.

For the wider 16/32-gram run, the same staged/final comparison found 1,120,341 staged pairs with 32,390/34,770 true links (93.1550% stage recall). The final cap-64 file retained 32,028 links (92.1139%). Thus 2,380 true links were absent after the first 64-per-source ranking cut and 362 were lost at final selection; the former are not proven raw proposal misses. Among the 2,742 final misses, 1,631 share at least one normalized name **and** address word, 466 share a name word only, and 645 share an address word only. These are token-overlap diagnostics; the broader-token run must still prove it can retrieve them efficiently.

A follow-up local diagnostic on the same 2,742 misses found 412 with identical order-independent informative-name signatures and 515 with identical compact joined-name keys. Because the groups may overlap and the experimental compact index is narrower than this diagnostic, 927 is only a loose upper bound on the new name-key channels' possible recoveries; no recall gain has been measured. The same report found 2,622 misses at exactly 32 emitted candidates from that target source, 97 below 32, and 23 above 32 due unused capacity from the other source. These counts refer to the **final per-source quota of 32**, not the 64-row staged shortlist. A true link can be absent from staging even when its source fills 32 final slots. Report: `D:\temp\amazon-ml-grams16-misses-namekeys.json` (outside Git).

The bounded broad-token run `amazon-ml-person-a-dev-broad-token` used the same 10,000 development IDs, all 10,320,219 train targets, K32/A16, 16 indexed/32 query character grams, and final cap 64. It raised word-pair frequency to 5,000 with at most 2,000 hits per pair and enabled two single-word probes per field up to frequency 512. Its final file retrieved **32,594/34,770 true links (93.7417%)**, 639,042 pairs, 0.977017 oracle macro F0.5, and 20.004 GiB **parent-process** peak RSS; fork-worker memory is not included. The Kaggle complete event was at about 1,952 seconds. The downloaded 10 staged chunks contained 1,238,656 pairs and 33,029 true links (94.9928% stage recall). Thus 1,741 links were absent after the first 64-per-source `_rank` cut and 435 staged links were removed before the final cap. The 1,741 may include proposed links ranked below 64; 94.9928% is **not** the raw proposal ceiling. A local residual audit found 2,176 final misses: 1,212 shared normalized name and address words, 425 name only, and 539 address only. Among those misses, 283 have equal name signatures and 365 have equal compact names, possibly overlapping; those keys alone cannot reach 98%. The final per-source count was exactly 32 for 2,137 misses. Reports and stage chunks remain outside Git under `D:\temp\amazon-ml-kernel-dev-broad-token-output` and `D:\temp\amazon-ml-broad-stage`.

The isolated `amazon-ml-person-a-dev-name-keys` run added sorted informative name tokens and compact website-style name keys to the 16/32-gram word-pair setup, keeping the same 10,000 IDs, full targets, K32/A16, and final cap 64. It retrieved **31,513/34,770 true links (90.6327%)**, 638,880 pairs, and 0.962342 oracle macro F0.5. Direct comparison with the otherwise matching 16/32-gram word-pair output found 448 newly recovered true links but 963 previously recovered links displaced, a net loss of 515 links. Thus the keys help some records but the current ranking/cap interaction makes this isolated change worse overall. Do not promote the keys without a better shortlist/ranker comparison. The pair and report artifacts are outside Git under `D:\temp\amazon-ml-kernel-dev-name-keys-output`.

An offline final-selection comparison on the **same saved 64-row-per-source broad-token stage** used `evaluate_stage_selection.py` with the seeded 10,000 development IDs, cap 64, and 32 reserved slots per source. The current score reproduced the original 32,594/34,770 links, 639,042 pairs, and 0.977017 oracle macro F0.5 exactly. Selecting by the mean of name and address cosine, without channel bonuses, kept **32,840/34,770 links (94.4492%)**, the same 639,042 pairs, and 0.980054 oracle macro F0.5. The 246-link gain is from final selection alone. Equal-weight cosine with quota 24 kept one additional true link but a slightly lower oracle macro F0.5 (0.980029); other tested rules were lower. Report: `D:\temp\amazon-ml-broad-stage-selection.json` (outside Git). This has not been validated on the holdout or used for full train/test output.

The isolated 8-million-feature hash run `amazon-ml-person-a-dev-hash8m` held all other broad-token settings fixed. At final cap 64 it retrieved **32,576/34,770 links (93.6900%)**, 639,089 pairs, and 0.976695 oracle macro F0.5. This is 18 fewer true links than the 1-million-feature broad-token run, so the larger hash space is not a measured improvement on development. Report: `D:\temp\amazon-ml-kernel-dev-hash8m-output\person-a-dev-report.json` (outside Git).

The combined `amazon-ml-person-a-broad-keys-stage128` run changed **several settings together** relative to broad-token: name keys, 128 rather than 64 staged rows per source, and wider cross-token probes. It retrieved 31,910/34,770 links (91.7745%) at the final 64-candidate cap, 639,672 pairs, and 0.966598 oracle macro F0.5. Its saved stage held 33,509/34,770 links (96.3733%) across 2,283,673 pairs. Reconstructing the original rank order from saved six-decimal scores, cutoff 64 per source held 33,145 links (95.3264%) and cutoff 128 held 33,509. Re-selecting the same 128-row stage with equal-weight name/address cosine and quota 32 kept **32,908/34,770 links (94.6448%)**, the same 639,672 final pairs, and 0.980903 oracle macro F0.5. This improves the run's final selection substantially, but its multiple proposal changes prevent attributing the stage gain to one change. Reports are outside Git under `D:\temp\amazon-ml-kernel-broad-keys-stage128-output` and `D:\temp\amazon-ml-broad-keys-stage128-{stage-report,curve,selection}.json`.

### Full labeled training result for earlier word-pair settings

All four private Kaggle training shards completed using word-pair retrieval, **K64/A32, 8 indexed/16 query grams**, and final cap 64. The exact-denominator aggregation checked 2,206,821 Source 1 records, 10,320,219 targets, and 7,638,365 true links. It found **6,927,452 true links (90.6929% recall)** in 141,236,138 candidate pairs. Complete-match-set coverage was 77.5613% across all Source 1 rows (76.2340% among nonempty truth rows), and the oracle macro F0.5 was 0.962982. Mean candidates per Source 1 were 63.9998, p95 was 64, and no Source 1 had zero candidates. Shards ran concurrently; their individual complete-event times were 5,689.26, 4,123.09, 4,029.58, and 6,585.76 seconds, so the slowest took about 109.8 minutes. Maximum reported **parent-process** peak RSS was 20.137 GiB, excluding fork-worker memory. This is the old word-pair configuration, not a full-data validation of the newer broad-token or balanced-ranking experiments. Aggregate report and per-shard reports/logs are outside Git under `D:\temp\person-a-full-train-word-pair-report.json` and `D:\temp\amazon-ml-train-word-pair-shard-{0,1,2,3}-report`.

### Full-test baseline artifact

All four private Kaggle test shards of the baseline K32/A16, cap-64 configuration completed. Their sorted long-form TSVs were merged outside Git to `D:\temp\amazon-ml-test-cap64.tsv`. The local `validate_candidate_long.py` command exited 0 over the merged 6.32 GB file: 110,336,665 unique sorted pairs, 1,732,543 represented test Source 1 IDs, cap 64, and all target IDs valid. One test Source 1 record has zero candidates and therefore no row in this long-form handoff; the final wide `candidate_pairs.tsv` must include every one of the 1,732,544 test Source 1 IDs, including an empty list for that record. This artifact is a validated fallback pending retrieval selection, not a matching result.
