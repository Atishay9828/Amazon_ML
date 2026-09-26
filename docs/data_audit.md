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

Recapping the same staged retrieval at cap 16 yielded 160,000 pairs, 83.1435% true-link recall, and 0.923140 oracle macro F0.5. Cap 64 yielded 636,560 pairs, 88.2773% recall, and 0.953626 oracle macro F0.5. A diagnostic cap 128 yielded 1,114,940 pairs, 89.2436% recall, and 0.957544 oracle macro F0.5. The small gain from cap 64 to 128 shows that the retrieval proposal, not only the final cap, misses many true links. The 50,000-row holdout remains untouched while development diagnostics continue.

Miss analysis of the diagnostic cap-128 table found 3,740 missed true links: 2,455 share at least one normalized name token **and** address token with their Source 1 record, 485 share a name token only, and 800 share an address token only. In 2,453 misses, that target source has fewer than its staged 64 candidates for the Source 1 record, indicating a proposal gap. In 1,287 misses, the target source is saturated at 64, so the staged per-source limit or ranking may be responsible. These categories are simple token overlap diagnostics, not proof that all such links can be recovered economically. The report with record examples stays outside Git at `D:\temp\amazon-ml-dev-misses.json`.

### Independent 50,000-row holdout check

The private Kaggle `amazon-ml-person-a-holdout-50k` run completed on the disjoint seeded 50,000 Source 1 selection, with the full 10,320,219 training targets and the same name K=32, address K=16, final cap 64 configuration. Its report recorded 172,581 true links, 3,182,916 candidate pairs, 88.1841% true-link recall, 72.45% complete-set coverage across all Source 1 records, and 0.951459 oracle macro F0.5. Mean candidates were 63.658 and p95 was 64. This is a retrieval check; it is not Person B's connected-component holdout or an actual matching score. The report was read once after development tuning, and no settings were chosen from its labels.

### Larger-neighbor development comparison

The private Kaggle `amazon-ml-person-a-dev-k64-a32` run used the **same** seeded 10,000 Source 1 IDs and complete 10,320,219 training targets as the baseline, with 64 name neighbors and 32 address neighbors per target source. At final cap 64 it retrieved 31,047/34,770 true links (89.2925%), emitted 640,000 pairs, and reached 0.957808 oracle macro F0.5. Peak reported process RSS was 20.113 GiB. The baseline at the same cap retrieved 30,694/34,770 links (88.2773%) and reached 0.953626 oracle macro F0.5. This gain is measurable but still leaves 3,723 true links unseen on development.

The subsequent cross-field token experiment `amazon-ml-person-a-dev-cross-token` kept K32/A16 and cap 64 on those same IDs/targets. It retrieved 30,965/34,770 true links (89.0567%), emitted 636,563 pairs, and reached 0.956852 oracle macro F0.5, with 19.934 GiB peak process RSS. A fresh missed-link audit found 3,805 true links absent from its final candidate table. This is an improvement over baseline but below the larger-neighbor experiment.

The combined `amazon-ml-person-a-dev-cross-token-k64-a32` run used cross-field tokens with K64/A32 at the same final cap 64. It retrieved 31,232/34,770 links (89.8246%), emitted 640,000 pairs, and reached 0.959808 oracle macro F0.5. Peak process RSS was 20.826 GiB; 3,538 true links remained unretrieved.

The same-field `name_pair`/`address_pair` run `amazon-ml-person-a-dev-word-pair` used K32/A16 and cap 64 on the same IDs/targets. It retrieved 31,356/34,770 links (90.1812%), emitted 636,937 pairs, and reached 0.962473 oracle macro F0.5, with 20.827 GiB peak process RSS. This is the best completed development result so far; it still misses 3,414 true links. A wider-character-gram run remains pending. A separate all-labeled-training evaluation is running in four Kaggle shards with word-pair retrieval and K64/A32; its result is not yet known.

### Full-test baseline artifact

All four private Kaggle test shards of the baseline K32/A16, cap-64 configuration completed. Their sorted long-form TSVs were merged outside Git to `D:\temp\amazon-ml-test-cap64.tsv`. The local `validate_candidate_long.py` command exited 0 over the merged 6.32 GB file: 110,336,665 unique sorted pairs, 1,732,543 represented test Source 1 IDs, cap 64, and all target IDs valid. One test Source 1 record has zero candidates and therefore no row in this long-form handoff; the final wide `candidate_pairs.tsv` must include every one of the 1,732,544 test Source 1 IDs, including an empty list for that record. This artifact is a validated fallback pending retrieval selection, not a matching result.
