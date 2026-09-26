# Person A handoff — 26 September 2026, 17:35 IST

## Repository state

- Branch: `feat/data-candidates`. Do not assume the local HEAD is pushed. `e6e12e7` already committed the learned shortlist ranker before the handoff request. `a79062f` then committed provisional `configs/final.json` and historical `configs/baseline.json`. Both commits are local only, two ahead of `origin/feat/data-candidates`. No push or history rewrite was made after the handoff request.
- The current working tree contains uncommitted edits to `candidates.py`, `rank_candidates.py`, `reverse_top1.py`, and `docs/person-a/runbook.md`; untracked `pipeline_config.py` and `run_pipeline.py`; and this handoff file. The ranktrain split and original ranker implementation are already in `e6e12e7`, not uncommitted.
- The full command `python -m unittest discover -s code/business_entity_resolution/tests -q` ran after the handoff request: **25 tests, one error**. `test_ranker_inference_reads_both_staged_sources` fails because the uncommitted config guard requires `config_hash` in its old fixture manifest. Per the user's instruction, no further commit was made. The suite had passed all 25 tests before these uncommitted config-guard edits.
- `run_pipeline.py` is unfinished and untested end to end. Do not use it to create the final candidate set yet. In particular, its baseline copy uses `read_bytes()` on the whole TSV, which is unsuitable for the full output; its optional external-model/reverse handling and resume checks need tests; and `package_kaggle_code.py` currently packages Python files but **not** the new `configs/` directory. The `pipeline_config.py` loader and all hash checks need fixture coverage. The full reverse Kaggle run uses provisional config hash `f42e3c9602539a9b73a687f38c356987825cbc15e30796bb4b38c93cb4273460`; changing any canonical config value invalidates that hash.
- The reviewer C1–C4 combined runs had no result in `../review/person-a-candidate-review.md` when last inspected. Its 96.74% figure is the union of R3 learned cap32 and reverse top1, **not** an end-to-end C1 measurement. No new combined dev run, frozen holdout, or full train/test candidate table for that final configuration has completed on this account. Candidate recall and oracle F0.5 are not Person B's matcher or leaderboard score.
- There is no named security-review script in this repository. Earlier commits used the full unit suite, `git diff --check`, staged file review, and private artifact checks. No push occurred for the two local commits.

## Kaggle kernels still RUNNING at the handoff check

Do not cancel these automatically. Their statuses were read from Kaggle CLI after the user asked to stop.

| Full slug | Work |
| --- | --- |
| `mridulnegi2005/amazon-ml-person-a-train-broad-balanced-0-of-4` | Full labeled train, shard 0/4; broad-token, 16/32 grams, stage64, balanced cap64 fallback. |
| `mridulnegi2005/amazon-ml-person-a-train-broad-balanced-1-of-4` | Same older fallback, train shard 1/4. |
| `mridulnegi2005/amazon-ml-person-a-test-sibling128-0-of-4` | Full test, shard 0/4; sibling-seeds2, stage128, balanced cap64 fallback. |
| `mridulnegi2005/amazon-ml-person-a-test-sibling128-1-of-4` | Same older fallback, test shard 1/4. |
| `mridulnegi2005/amazon-ml-person-a-reverse-train-full` | Same-split train Source 1 char-3 TF-IDF reverse top1 over all train targets; 4 workers, DF cap 0.002, provisional config hash above. It writes `reverse-train.tsv` and `.meta.json` on Kaggle when complete. |

Completed recent kernel: `mridulnegi2005/amazon-ml-person-a-reverse-smoke`. It used the full 2,206,821-row train Source 1 index and only the first 100,000 targets **from each** of Source 2 and 3. Index: 78.04 s. Total: 110.77 s. It wrote 191,810 reverse pairs; this is a timing sample, not recall. Report: `D:\temp\amazon-ml-reverse-smoke-output\reverse-smoke.tsv.meta.json`. No Kaggle kernels were started after the stop request.

Other completed kernels and metrics are in `docs/data_audit.md` and `docs/person-a/todo.md`. The private code dataset is `mridulnegi2005/amazon-ml-person-a-code`; private official data dataset is `mridulnegi2005/amazon-ml-challenge-2026-private-input`. The current uploaded code ZIP contains the uncommitted config-guard source snapshot from before `run_pipeline.py` was created. It is not a verified final pipeline.

## Finished measured experiments

All dev numbers use the seeded 10,000 train Source 1 IDs and all 10,320,219 train targets, with 34,770 true links, except where noted.

| Experiment/configuration | True links retrieved | Candidate pairs | Oracle macro F0.5 | Status and artifact |
| --- | ---: | ---: | ---: | --- |
| Original word-pair K32/A16, 8/16 grams, cap64 | 31,356/34,770 (90.1812%) | About 640,000 | 0.962473 | Completed dev; `D:\temp\amazon-ml-kernel-dev-word-pair-output` |
| Broader word pairs and token probes, 16/32 grams, stage64, cap64 `_rank` | 32,594/34,770 (93.7417%) | 639,042 | 0.977017 | Completed dev; `D:\temp\amazon-ml-kernel-dev-broad-token-output` |
| Same broad stage, offline balanced cap64 | 32,840/34,770 (94.4492%) | 639,042 | 0.980054 | Completed dev reconstruction; `D:\temp\amazon-ml-broad-stage-selection.json` |
| Sibling-seeds2, stage2048, balanced cap64 | 33,139/34,770 (95.3092%) | 639,876 | 0.982735 | Completed dev; `D:\temp\amazon-ml-sibling2-output`; staged ceiling 33,517/34,770 |
| Sibling stage128, offline balanced cap96 | 33,284/34,770 (95.7262%) | 958,624 | 0.984159 | Dev reconstruction only; `D:\temp\amazon-ml-sibling2-stage128-cap96-selection.json` |
| Broad/balanced 50,000-row holdout, cap64 | 162,720/172,581 (94.2862%) | 3,194,567 | 0.977965 | Completed separate holdout; `D:\temp\amazon-ml-holdout-broad-balanced-output` |
| Sibling128/balanced 50,000-row holdout, cap64 | 163,955/172,581 (95.0018%) | 3,199,500 | 0.980287 | Completed separate holdout; `D:\temp\amazon-ml-holdout-sibling128-output\person-a-holdout-report.json` |
| Old word-pair K64/A32, 8/16 grams, cap64 on **all 2,206,821 labeled train Source 1** | 6,927,452/7,638,365 (90.6929%) | 141,236,138 | 0.962982 | Completed four-shard aggregation; `D:\temp\person-a-full-train-word-pair-report.json` |

Additional controlled dev results: wider word probes stage128/final 33,331/33,026 links; wider 32/64 character grams stage128/final 33,256/32,975; name keys stage128/final 33,430/32,844; stable-hash target grams at stage64/final after offline balanced selection 33,088/32,900; 8-million hash features 32,576 final (18 below broad-token). See `docs/data_audit.md` and the artifact inventory below for reports. The independent review reports higher learned-ranker/reverse results, but they were not reproduced on this branch.

The old baseline full-test long-form file `D:\temp\amazon-ml-test-cap64.tsv` was validated: 110,336,665 unique sorted pairs, 1,732,543 Source 1 IDs represented. The one Source 1 ID with no candidates must appear with an empty list in Person C's wide `candidate_pairs.tsv`. This file predates the newer configurations and is not the final learned-ranker candidate set.

## Local artifact inventory

The appendix lists each `D:\temp` Person A artifact found at handoff. A `kernel-*` directory contains a Kaggle launcher and its metadata; an `*-output` directory contains downloaded private Kaggle log, report, and sometimes candidate TSV; a `*-stage` directory contains downloaded compressed per-source candidate chunks and a manifest. Names with `word-pair`, `broad`, `sibling`, `grams`, `namekeys`, `hash`, `min-df`, or `cross-token` identify the experimental configuration described above and in `docs/data_audit.md`. A JSON named `*-selection`, `*-curve`, `*-misses`, or `*-report` holds the named offline selection, cutoff, missed-link diagnostic, or metric report. Local `*.work` directories hold checkpoint chunks and indexes for the named bounded benchmark. Some older scratch paths may be partial; use the explicit completion reports above before treating one as a result. The `review/` directory is teammate-provided and is not part of this inventory.

<!-- Exact outside-Git path inventory follows. -->

### Exact paths found under `D:\temp`

- `D:\Temp\amazon-ml-alt-progress` — directory of logs, reports, TSVs, or staged checkpoint chunks; Person A scratch artifact; read its report, manifest, or script before reuse.
- `D:\Temp\amazon-ml-benchmark-1m.tsv` — long-form candidate TSV; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-benchmark-1m.work` — directory of logs, reports, TSVs, or staged checkpoint chunks; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-benchmark-500k.tsv` — long-form candidate TSV; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-benchmark-500k.work` — directory of logs, reports, TSVs, or staged checkpoint chunks; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-broad-balanced-cap16.json` — JSON report/diagnostic; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-broad-balanced-cap32.json` — JSON report/diagnostic; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-broad-keys-stage128-curve.json` — JSON report/diagnostic; combined name keys and wider cross probes, stage128 diagnostic.
- `D:\Temp\amazon-ml-broad-keys-stage128-selection.json` — JSON report/diagnostic; combined name keys and wider cross probes, stage128 diagnostic.
- `D:\Temp\amazon-ml-broad-keys-stage128-stage` — directory of logs, reports, TSVs, or staged checkpoint chunks; combined name keys and wider cross probes, stage128 diagnostic.
- `D:\Temp\amazon-ml-broad-keys-stage128-stage-report.json` — JSON report/diagnostic; combined name keys and wider cross probes, stage128 diagnostic.
- `D:\Temp\amazon-ml-broad-misses.json` — JSON report/diagnostic; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-broad-namekeys128-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token plus name keys, stage128, balanced cap64.
- `D:\Temp\amazon-ml-broad-namekeys128-stage` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token plus name keys, stage128, balanced cap64.
- `D:\Temp\amazon-ml-broad-namekeys128-stage-report.json` — JSON report/diagnostic; broad-token plus name keys, stage128, balanced cap64.
- `D:\Temp\amazon-ml-broad-stage` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-broad-stage-curve.json` — JSON report/diagnostic; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-broad-stage-report.json` — JSON report/diagnostic; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-broad-stage-selection.json` — JSON report/diagnostic; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-broad-stage128-selection.json` — JSON report/diagnostic; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-broad-wordprobes128-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token with 6/8 cross-field word probes, stage128, balanced cap64.
- `D:\Temp\amazon-ml-broad-wordprobes128-stage` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token with 6/8 cross-field word probes, stage128, balanced cap64.
- `D:\Temp\amazon-ml-broad-wordprobes128-stage-report.json` — JSON report/diagnostic; broad-token with 6/8 cross-field word probes, stage128, balanced cap64.
- `D:\Temp\amazon-ml-candidate-50k.tsv` — long-form candidate TSV; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-candidate-50k.work` — directory of logs, reports, TSVs, or staged checkpoint chunks; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-candidate-indexed-50k.tsv` — long-form candidate TSV; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-candidate-indexed-50k.work` — directory of logs, reports, TSVs, or staged checkpoint chunks; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-candidate-smoke.tsv` — long-form candidate TSV; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-candidate-smoke.work` — directory of logs, reports, TSVs, or staged checkpoint chunks; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-code-upload` — scratch directory; private Kaggle source ZIP and dataset metadata; snapshot includes uncommitted hash-guard code but no run_pipeline.py.
- `D:\Temp\amazon-ml-cross-token-misses.json` — JSON report/diagnostic; cross-field token retrieval experiment.
- `D:\Temp\amazon-ml-dev-cap16-report.json` — JSON report/diagnostic; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-dev-cap16.tsv` — long-form candidate TSV; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-dev-cap64-report.json` — JSON report/diagnostic; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-dev-cap64.tsv` — long-form candidate TSV; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-dev-misses.json` — JSON report/diagnostic; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-dev-stage128-report.json` — JSON report/diagnostic; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-dev-stage128.tsv` — long-form candidate TSV; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-grams16-misses-namekeys.json` — JSON report/diagnostic; word-pair with 16 indexed and 32 query grams.
- `D:\Temp\amazon-ml-grams16-misses.json` — JSON report/diagnostic; word-pair with 16 indexed and 32 query grams.
- `D:\Temp\amazon-ml-grams16-stage` — directory of logs, reports, TSVs, or staged checkpoint chunks; word-pair with 16 indexed and 32 query grams.
- `D:\Temp\amazon-ml-grams16-stage-report.json` — JSON report/diagnostic; word-pair with 16 indexed and 32 query grams.
- `D:\Temp\amazon-ml-grams32-stage128-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token with 32 indexed and 64 query grams, stage128, balanced cap64.
- `D:\Temp\amazon-ml-grams32-stage128-stage` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token with 32 indexed and 64 query grams, stage128, balanced cap64.
- `D:\Temp\amazon-ml-grams32-stage128-stage-report.json` — JSON report/diagnostic; broad-token with 32 indexed and 64 query grams, stage128, balanced cap64.
- `D:\Temp\amazon-ml-holdout-broad-balanced-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token 50,000-row holdout, stage64, balanced cap64.
- `D:\Temp\amazon-ml-holdout-sibling128-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; sibling-seeds2 50,000-row holdout, stage128, balanced cap64.
- `D:\Temp\amazon-ml-inspect-misses.py` — local helper script; missed-link diagnostic for the named dev experiment.
- `D:\Temp\amazon-ml-kaggle-upload` — scratch directory; private official-data Kaggle upload staging; contains challenge data, keep outside Git.
- `D:\Temp\amazon-ml-kernel-broad-keys-stage128-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; combined name keys and wider cross probes, stage128 diagnostic.
- `D:\Temp\amazon-ml-kernel-dev` — Kaggle launcher and kernel metadata; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-kernel-dev-broad-keys-stage128` — Kaggle launcher and kernel metadata; combined name keys and wider cross probes, stage128 diagnostic.
- `D:\Temp\amazon-ml-kernel-dev-broad-namekeys128` — Kaggle launcher and kernel metadata; broad-token plus name keys, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-dev-broad-namekeys128-clean` — Kaggle launcher and kernel metadata; broad-token plus name keys, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-dev-broad-token` — Kaggle launcher and kernel metadata; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-kernel-dev-broad-token-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token proposal baseline or offline cap/depth comparison; see file name.
- `D:\Temp\amazon-ml-kernel-dev-broad-wordprobes128` — Kaggle launcher and kernel metadata; broad-token with 6/8 cross-field word probes, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-dev-broad-wordprobes128-clean` — Kaggle launcher and kernel metadata; broad-token with 6/8 cross-field word probes, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-dev-cross-token` — Kaggle launcher and kernel metadata; cross-field token retrieval experiment.
- `D:\Temp\amazon-ml-kernel-dev-cross-token-k64a32` — Kaggle launcher and kernel metadata; cross-field token retrieval experiment.
- `D:\Temp\amazon-ml-kernel-dev-cross-token-k64a32-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; cross-field token retrieval experiment.
- `D:\Temp\amazon-ml-kernel-dev-cross-token-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; cross-field token retrieval experiment.
- `D:\Temp\amazon-ml-kernel-dev-grams32-stage128` — Kaggle launcher and kernel metadata; broad-token with 32 indexed and 64 query grams, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-dev-hash8m` — Kaggle launcher and kernel metadata; broad-token with 8-million hashed features.
- `D:\Temp\amazon-ml-kernel-dev-hash8m-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token with 8-million hashed features.
- `D:\Temp\amazon-ml-kernel-dev-k64a32` — Kaggle launcher and kernel metadata; word-pair or cross-token retrieval with K64/A32.
- `D:\Temp\amazon-ml-kernel-dev-k64a32-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; word-pair or cross-token retrieval with K64/A32.
- `D:\Temp\amazon-ml-kernel-dev-min-df2` — Kaggle launcher and kernel metadata; broad-token with minimum indexed gram document frequency 2.
- `D:\Temp\amazon-ml-kernel-dev-min-df2-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token with minimum indexed gram document frequency 2.
- `D:\Temp\amazon-ml-kernel-dev-name-keys` — Kaggle launcher and kernel metadata; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-kernel-dev-name-keys-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-kernel-dev-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-kernel-dev-sibling2` — Kaggle launcher and kernel metadata; sibling-seeds2 dev, stage2048 or offline stage128, balanced final selection.
- `D:\Temp\amazon-ml-kernel-dev-stable-grams` — Kaggle launcher and kernel metadata; broad-token with stable-hash target gram choice, 16/32 grams.
- `D:\Temp\amazon-ml-kernel-dev-stable-grams-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token with stable-hash target gram choice, 16/32 grams.
- `D:\Temp\amazon-ml-kernel-dev-stage2048` — Kaggle launcher and kernel metadata; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-kernel-dev-stage2048-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; older dev launcher/output or offline dev cap/depth diagnostic; see report/manifest for exact flags.
- `D:\Temp\amazon-ml-kernel-dev-word-pair` — Kaggle launcher and kernel metadata; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\amazon-ml-kernel-dev-word-pair-grams16` — Kaggle launcher and kernel metadata; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\amazon-ml-kernel-dev-word-pair-grams16-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\amazon-ml-kernel-dev-word-pair-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\amazon-ml-kernel-holdout` — Kaggle launcher and kernel metadata; older holdout launcher or output; see the report inside for exact flags.
- `D:\Temp\amazon-ml-kernel-holdout-broad-balanced` — Kaggle launcher and kernel metadata; broad-token 50,000-row holdout, stage64, balanced cap64.
- `D:\Temp\amazon-ml-kernel-holdout-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; older holdout launcher or output; see the report inside for exact flags.
- `D:\Temp\amazon-ml-kernel-holdout-sibling128` — Kaggle launcher and kernel metadata; sibling-seeds2 50,000-row holdout, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-smoke` — Kaggle launcher and kernel metadata; bounded smoke sample/launcher; no valid full-target recall.
- `D:\Temp\amazon-ml-kernel-smoke-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; bounded smoke sample/launcher; no valid full-target recall.
- `D:\Temp\amazon-ml-kernel-test-0` — Kaggle launcher and kernel metadata; older baseline full-test shard, K32/A16, cap64.
- `D:\Temp\amazon-ml-kernel-test-1` — Kaggle launcher and kernel metadata; older baseline full-test shard, K32/A16, cap64.
- `D:\Temp\amazon-ml-kernel-test-2` — Kaggle launcher and kernel metadata; older baseline full-test shard, K32/A16, cap64.
- `D:\Temp\amazon-ml-kernel-test-3` — Kaggle launcher and kernel metadata; older baseline full-test shard, K32/A16, cap64.
- `D:\Temp\amazon-ml-kernel-test-sibling128-0` — Kaggle launcher and kernel metadata; sibling-seeds2 test fallback, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-test-sibling128-1` — Kaggle launcher and kernel metadata; sibling-seeds2 test fallback, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-test-sibling128-2` — Kaggle launcher and kernel metadata; sibling-seeds2 test fallback, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-test-sibling128-3` — Kaggle launcher and kernel metadata; sibling-seeds2 test fallback, stage128, balanced cap64.
- `D:\Temp\amazon-ml-kernel-test-word-pair-0` — Kaggle launcher and kernel metadata; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\amazon-ml-kernel-test-word-pair-1` — Kaggle launcher and kernel metadata; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\amazon-ml-kernel-test-word-pair-2` — Kaggle launcher and kernel metadata; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\amazon-ml-kernel-test-word-pair-3` — Kaggle launcher and kernel metadata; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\amazon-ml-kernel-train-broad-balanced-0` — Kaggle launcher and kernel metadata; broad-token train fallback, 16/32 grams, stage64, balanced cap64.
- `D:\Temp\amazon-ml-kernel-train-broad-balanced-1` — Kaggle launcher and kernel metadata; broad-token train fallback, 16/32 grams, stage64, balanced cap64.
- `D:\Temp\amazon-ml-kernel-train-sibling128-0` — Kaggle launcher and kernel metadata; prepared sibling-seeds2 train shard launcher, stage128, balanced cap96; not launched.
- `D:\Temp\amazon-ml-kernel-train-sibling128-1` — Kaggle launcher and kernel metadata; prepared sibling-seeds2 train shard launcher, stage128, balanced cap96; not launched.
- `D:\Temp\amazon-ml-kernel-train-word-pair-0` — Kaggle launcher and kernel metadata; original word-pair full train, K64/A32, 8/16 grams, cap64.
- `D:\Temp\amazon-ml-kernel-train-word-pair-1` — Kaggle launcher and kernel metadata; original word-pair full train, K64/A32, 8/16 grams, cap64.
- `D:\Temp\amazon-ml-kernel-train-word-pair-2` — Kaggle launcher and kernel metadata; original word-pair full train, K64/A32, 8/16 grams, cap64.
- `D:\Temp\amazon-ml-kernel-train-word-pair-3` — Kaggle launcher and kernel metadata; original word-pair full train, K64/A32, 8/16 grams, cap64.
- `D:\Temp\amazon-ml-local-2k-limited.work` — directory of logs, reports, TSVs, or staged checkpoint chunks; bounded local candidate benchmark/checkpoint; not full-data recall.
- `D:\Temp\amazon-ml-make-smoke-sample.py` — local helper script; bounded smoke sample/launcher; no valid full-target recall.
- `D:\Temp\amazon-ml-person-a-audit.json` — JSON report/diagnostic; full train/test input audit with counts and hashes.
- `D:\Temp\amazon-ml-prepare-kaggle-upload.py` — local helper script; private official-data Kaggle upload staging; contains challenge data, keep outside Git.
- `D:\Temp\amazon-ml-profile.prof` — local artifact; local profiling trace from bounded candidate run.
- `D:\Temp\amazon-ml-reverse-smoke` — scratch directory; reverse top1 timing sample, full train S1 index and first 100,000 targets per source.
- `D:\Temp\amazon-ml-reverse-smoke-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; reverse top1 timing sample, full train S1 index and first 100,000 targets per source.
- `D:\Temp\amazon-ml-reverse-train-full` — scratch directory; provisional train reverse top1, DF 0.002, 4 workers, config hash f42e3c96; launcher only until Kaggle output is downloaded.
- `D:\Temp\amazon-ml-sibling2-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; sibling-seeds2 dev, stage2048 or offline stage128, balanced final selection.
- `D:\Temp\amazon-ml-sibling2-progress` — directory of logs, reports, TSVs, or staged checkpoint chunks; sibling-seeds2 dev, stage2048 or offline stage128, balanced final selection.
- `D:\Temp\amazon-ml-sibling2-stage` — directory of logs, reports, TSVs, or staged checkpoint chunks; sibling-seeds2 dev, stage2048 or offline stage128, balanced final selection.
- `D:\Temp\amazon-ml-sibling2-stage-report.json` — JSON report/diagnostic; sibling-seeds2 dev, stage2048 or offline stage128, balanced final selection.
- `D:\Temp\amazon-ml-sibling2-stage128-cap128-selection.json` — JSON report/diagnostic; sibling-seeds2 dev, stage2048 or offline stage128, balanced final selection.
- `D:\Temp\amazon-ml-sibling2-stage128-cap256-selection.json` — JSON report/diagnostic; sibling-seeds2 dev, stage2048 or offline stage128, balanced final selection.
- `D:\Temp\amazon-ml-sibling2-stage128-cap96-selection.json` — JSON report/diagnostic; sibling-seeds2 dev, stage2048 or offline stage128, balanced final selection.
- `D:\Temp\amazon-ml-sibling2-stage128-selection.json` — JSON report/diagnostic; sibling-seeds2 dev, stage2048 or offline stage128, balanced final selection.
- `D:\Temp\amazon-ml-smoke-sample` — scratch directory; bounded smoke sample/launcher; no valid full-target recall.
- `D:\Temp\amazon-ml-stable-grams-selection.json` — JSON report/diagnostic; broad-token with stable-hash target gram choice, 16/32 grams.
- `D:\Temp\amazon-ml-stable-grams-stage` — directory of logs, reports, TSVs, or staged checkpoint chunks; broad-token with stable-hash target gram choice, 16/32 grams.
- `D:\Temp\amazon-ml-stable-grams-stage-report.json` — JSON report/diagnostic; broad-token with stable-hash target gram choice, 16/32 grams.
- `D:\Temp\amazon-ml-stage2048` — scratch directory; Person A scratch artifact; read its report, manifest, or script before reuse.
- `D:\Temp\amazon-ml-stage2048-curve.json` — JSON report/diagnostic; Person A scratch artifact; read its report, manifest, or script before reuse.
- `D:\Temp\amazon-ml-stage2048-miss-fields.json` — JSON report/diagnostic; Person A scratch artifact; read its report, manifest, or script before reuse.
- `D:\Temp\amazon-ml-stage2048-selection.json` — JSON report/diagnostic; Person A scratch artifact; read its report, manifest, or script before reuse.
- `D:\Temp\amazon-ml-test-cap64.tsv` — long-form candidate TSV; merged and validated older baseline full-test long-form candidate TSV.
- `D:\Temp\amazon-ml-test-shard-0-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; older baseline full-test shard, K32/A16, cap64.
- `D:\Temp\amazon-ml-test-shard-1-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; older baseline full-test shard, K32/A16, cap64.
- `D:\Temp\amazon-ml-test-shard-2-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; older baseline full-test shard, K32/A16, cap64.
- `D:\Temp\amazon-ml-test-shard-3-output` — directory of logs, reports, TSVs, or staged checkpoint chunks; older baseline full-test shard, K32/A16, cap64.
- `D:\Temp\amazon-ml-train-word-pair-shard-0-report` — scratch directory; original word-pair full train, K64/A32, 8/16 grams, cap64.
- `D:\Temp\amazon-ml-train-word-pair-shard-1-report` — scratch directory; original word-pair full train, K64/A32, 8/16 grams, cap64.
- `D:\Temp\amazon-ml-train-word-pair-shard-2-report` — scratch directory; original word-pair full train, K64/A32, 8/16 grams, cap64.
- `D:\Temp\amazon-ml-train-word-pair-shard-3-report` — scratch directory; original word-pair full train, K64/A32, 8/16 grams, cap64.
- `D:\Temp\amazon-ml-word-pair-misses.json` — JSON report/diagnostic; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\amazon-ml-wordpair-stage` — directory of logs, reports, TSVs, or staged checkpoint chunks; original word-pair dev, K32/A16, 8/16 grams unless grams16 is named.
- `D:\Temp\person-a-full-train-word-pair-report.json` — JSON report/diagnostic; original word-pair full train, K64/A32, 8/16 grams, cap64.

### Kaggle output identities

These are the exact private kernel slugs recorded by local launcher metadata. A launcher directory alone does not prove its kernel was pushed or completed. The five running slugs and the completed reverse smoke are verified above. The sibling128 test shard 2/3 and sibling128 train shard 0/1 launchers were prepared locally but not pushed. For other slugs, check Kaggle status and the downloaded report before relying on outputs.

- `mridulnegi2005/amazon-ml-person-a-broad-keys-stage128` — amazon-ml-kernel-dev-broad-keys-stage128 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-cpu-smoke-benchmark` — amazon-ml-kernel-smoke configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-10k` — amazon-ml-kernel-dev configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-broad-namekeys128` — amazon-ml-kernel-dev-broad-namekeys128 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-broad-namekeys128` — amazon-ml-kernel-dev-broad-namekeys128-clean configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-broad-token` — amazon-ml-kernel-dev-broad-token configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-broad-wordprobes128` — amazon-ml-kernel-dev-broad-wordprobes128 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-broad-wordprobes128` — amazon-ml-kernel-dev-broad-wordprobes128-clean configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-cross-token` — amazon-ml-kernel-dev-cross-token configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-cross-token-k64-a32` — amazon-ml-kernel-dev-cross-token-k64a32 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-grams32-stage128` — amazon-ml-kernel-dev-grams32-stage128 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-hash8m` — amazon-ml-kernel-dev-hash8m configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-k64-a32` — amazon-ml-kernel-dev-k64a32 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-min-df2` — amazon-ml-kernel-dev-min-df2 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-name-keys` — amazon-ml-kernel-dev-name-keys configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-sibling2` — amazon-ml-kernel-dev-sibling2 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-stable-grams` — amazon-ml-kernel-dev-stable-grams configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-stage2048` — amazon-ml-kernel-dev-stage2048 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-word-pair` — amazon-ml-kernel-dev-word-pair configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-dev-word-pair-grams16` — amazon-ml-kernel-dev-word-pair-grams16 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-holdout-50k` — amazon-ml-kernel-holdout configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-holdout-broad-balanced` — amazon-ml-kernel-holdout-broad-balanced configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-holdout-sibling128` — amazon-ml-kernel-holdout-sibling128 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-reverse-smoke` — amazon-ml-reverse-smoke configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-reverse-train-full` — amazon-ml-reverse-train-full configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-shard-0-of-4` — amazon-ml-kernel-test-0 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-shard-1-of-4` — amazon-ml-kernel-test-1 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-shard-2-of-4` — amazon-ml-kernel-test-2 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-shard-3-of-4` — amazon-ml-kernel-test-3 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-sibling128-0-of-4` — amazon-ml-kernel-test-sibling128-0 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-sibling128-1-of-4` — amazon-ml-kernel-test-sibling128-1 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-sibling128-2-of-4` — amazon-ml-kernel-test-sibling128-2 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-sibling128-3-of-4` — amazon-ml-kernel-test-sibling128-3 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-word-pair-shard-0-of-4` — amazon-ml-kernel-test-word-pair-0 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-word-pair-shard-1-of-4` — amazon-ml-kernel-test-word-pair-1 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-word-pair-shard-2-of-4` — amazon-ml-kernel-test-word-pair-2 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-test-word-pair-shard-3-of-4` — amazon-ml-kernel-test-word-pair-3 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-train-broad-balanced-0-of-4` — amazon-ml-kernel-train-broad-balanced-0 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-train-broad-balanced-1-of-4` — amazon-ml-kernel-train-broad-balanced-1 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-train-sibling128-0-of-4` — amazon-ml-kernel-train-sibling128-0 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-train-sibling128-1-of-4` — amazon-ml-kernel-train-sibling128-1 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-train-word-pair-shard-0-of-4` — amazon-ml-kernel-train-word-pair-0 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-train-word-pair-shard-1-of-4` — amazon-ml-kernel-train-word-pair-1 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-train-word-pair-shard-2-of-4` — amazon-ml-kernel-train-word-pair-2 configuration; output is private under `/kaggle/working` if the kernel was run.
- `mridulnegi2005/amazon-ml-person-a-train-word-pair-shard-3-of-4` — amazon-ml-kernel-train-word-pair-3 configuration; output is private under `/kaggle/working` if the kernel was run.
