# Person A — evidence-based tracker

Status date: 26 September 2026 IST. `[x]` means the evidence below was checked; unchecked work has not been completed. Attribution is explicit.

## Verified starting context (not Person A implementation)

- [x] Codex verified `origin/main` contains shared context but no candidate code; `origin/docs/team-workstreams` contains Atishay's planning/workstream PR. Evidence: `git log --all --oneline` and `git show origin/docs/team-workstreams:docs/workstreams/person-a.md` (26 Sep).
- [x] Codex verified official TSVs exist outside the repository at `../student_resource/dataset/`. Evidence: file enumeration and header reads (26 Sep). No challenge rows were copied into Git.
- [x] Codex previously counted approximately 2.21m train and 1.73m test Source 1 rows and found 28.31% exact normalized name-or-address coverage on a seeded 5,000-record training sample. This is a preliminary sample, **not** candidate recall or a challenge score. Full audit and reproducible sample script are pending.

## Person A work

- [x] Implement and run strict streaming TSV loader and complete full-data audit; record commands, counts, hashes, and anomalies in `docs/data_audit.md`. Evidence: full audit exited 0; seven input hashes and counts recorded there (Codex, 26 Sep).
- [x] Implement and test conservative name/address normalization, including empty and Unicode inputs. Evidence: `test_normalization_preserves_digits_and_avoids_blank_blocks` passed (Codex, 26 Sep).
- [x] Implement exact-name, exact-address, rare-token, and character TF-IDF candidate retrieval with the fixed five-column contract and deterministic cap. Evidence: `src/candidates.py`; invented-fixture generation and repeat run passed. Full-scale quality remains unmeasured (Codex, 26 Sep).
- [x] Run invented-fixture tests for parsing, deduplication, candidate IDs, scores, France, no-candidate entities, and ordering. Evidence: `python -m unittest discover -s code/business_entity_resolution/tests -v` — 5 passed (Codex, 26 Sep).
- [x] Benchmark a bounded local sample, including one million targets per source and peak RAM. Evidence: `candidates.py --split train --limit-source1 100 --limit-targets 1000000 --query-chunk 100 --matrix-chunk 100000 --cap 32`; 3.20 GiB peak, 346.30 s/367.62 s source indexing (Codex, 26 Sep). Full-source resource behavior remains pending.
- [ ] Tune caps on Person B's development split when available; record true-link recall, complete-set coverage, volume, oracle macro F0.5, missed-link categories, and chosen configuration.
- [ ] Execute full train/test candidate generation in private compute; check files against the contract and provide them to B/C without committing them.
- [ ] Open a PR to main for Atishay with code, run commands, dependency versions, actual measurements, and remaining risks. Atishay handles merge and portal submission.

## Run log

| Date/time IST | Actor | Action | Evidence/result | Status |
| --- | --- | --- | --- | --- |
| 26 Sep 2026 | Codex | Created `feat/data-candidates` from `origin/main` | `git switch -c feat/data-candidates origin/main` | Done |
| 26 Sep 2026 | Codex | Audited all seven official TSVs | `python code/business_entity_resolution/src/data.py --data-root '..\student_resource\dataset'`; 2,206,821 train S1, 7,638,365 links; see `docs/data_audit.md` | Done |
| 26 Sep 2026 | Codex | Ran invented-fixture suite | 5 tests passed, including deterministic repeat, France, and oracle singleton scoring | Done |
| 26 Sep 2026 | Codex | Improved candidate retrieval on a biased 1,000-S1 scratch corpus | Rare address tokens: 0.9861 true-link recall at cap 64; cap 32: 0.9821. Transliterated/vote-ranked cap 32: 0.9861. All optimistic; see `docs/data_audit.md` | Exploratory only |
| 26 Sep 2026 | Codex | Indexed 1,000,000 official targets in each source on local CPU | Source 2 346.30 s, Source 3 367.62 s; 3.20 GiB peak working set; 3,200 pairs from 100 S1 rows | Done, resource benchmark only |
| 26 Sep 2026 | Codex | Uploaded official TSVs and A-only source to private Kaggle datasets using Kaggle CLI | `kaggle datasets files mridulnegi2005/amazon-ml-challenge-2026-private-input` lists seven TSVs; code dataset lists four Python files | Done; datasets private |
| 26 Sep 2026 | Codex | Started private Kaggle CPU smoke and seeded dev kernels | `kaggle kernels push -p D:\temp\amazon-ml-kernel-smoke`; `kaggle kernels push -p D:\temp\amazon-ml-kernel-dev` | Running; results pending |

Record new entries immediately after each verified run or commit. Never infer completion from a teammate's plan or an unrun command.
