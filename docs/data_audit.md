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
- 123,247 true singletons (5.59% of Source 1). Maximum true matches per Source 1 is 11.
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
