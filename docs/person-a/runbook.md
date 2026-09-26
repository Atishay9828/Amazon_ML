# Person A candidate generation runbook

This is the long-form candidate table handed to Person B for scoring. The
official wide-format `candidate_pairs.tsv` and `matching_results.tsv` are
assembled by Person C. No official data, candidate output, or credentials
belong in Git.

## Local commands

From the repository root, with Python 3.12+ and the A-owned pinned
dependencies in `docs/person-a/requirements.txt`. These match the versions
printed by the completed private Kaggle CPU smoke (Python 3.12.13; NumPy
2.0.2; SciPy 1.16.3; scikit-learn 1.6.1; AnyAscii 0.3.3; psutil 7.1.0):

```powershell
python -m pip install -r docs/person-a/requirements.txt
python -m unittest discover -s code/business_entity_resolution/tests -v
python code/business_entity_resolution/src/data.py --data-root '..\student_resource\dataset' > D:\temp\amazon-ml-person-a-audit.json
python code/business_entity_resolution/src/candidates.py --data-root '..\student_resource\dataset' --split train --out D:\temp\person-a-dev.tsv --work-dir D:\temp\person-a-dev.work --sample-split dev --cap 32 --report D:\temp\person-a-dev-report.json
python code/business_entity_resolution/src/candidates.py --data-root '..\student_resource\dataset' --split train --out D:\temp\person-a-holdout.tsv --work-dir D:\temp\person-a-holdout.work --sample-split holdout --cap 32 --report D:\temp\person-a-holdout-report.json
```

The seeded development and holdout selections are disjoint (seed 20260926,
10,000 and 50,000 Source 1 rows). Both search the **complete** target sources.
For full handoff files, omit `--sample-split` and use `--split train` or
`--split test`. `--workers 4` uses shared-memory fork workers on Kaggle Linux;
Windows falls back to one worker. The work directory contains resumable
compressed per-source chunks. Keep it when changing only `--cap`, because the
same retrieved rows can be reassembled at caps 16, 32, or 64. Use a fresh work
directory if any retrieval setting or input changes; the manifest guards this.
The CLI also exposes `--indexed-grams`, `--query-grams`, and `--max-probe-df`
for bounded character-gram retrieval experiments; defaults are 8, 16, and
2,000. The private `amazon-ml-person-a-dev-word-pair-grams16` run tests 16
indexed and 32 query grams while keeping the other settings and sample fixed.

For a full run that may exceed one Kaggle session, use `--shard-count 4` and
`--shard-index 0`, `1`, `2`, or `3` in four separate kernels. Shards are
contiguous slices of sorted Source 1 IDs. All four outputs have the same
header, and `merge_candidate_shards.py` checks order while concatenating them:

```powershell
python code/business_entity_resolution/src/merge_candidate_shards.py --out D:\temp\person-a-test-cap64.tsv D:\temp\test-shard-0.tsv D:\temp\test-shard-1.tsv D:\temp\test-shard-2.tsv D:\temp\test-shard-3.tsv
python code/business_entity_resolution/src/recap_candidates.py --in D:\temp\person-a-test-cap64.tsv --out D:\temp\person-a-test-cap32.tsv --cap 32
python code/business_entity_resolution/src/validate_candidate_long.py --input D:\temp\person-a-test-cap32.tsv --data-root '..\student_resource\dataset' --split test --cap 32
```

The cap-64 output retains the candidates needed for lower caps with the same
ranking and source quotas. For representative train evaluation after reducing
cap 64 to 32 or 16, run:

```powershell
python code/business_entity_resolution/src/evaluate_candidate_file.py --input D:\temp\person-a-dev-cap32.tsv --data-root '..\student_resource\dataset' --sample-split dev --out D:\temp\person-a-dev-cap32-report.json
```

For full labeled training in four sorted shards, each Kaggle `train-N-of-4`
kernel writes `person-a-train-N-of-4-report.json` and a log. Download all four
reports/logs and aggregate with exact link and entity denominators:

```powershell
python code/business_entity_resolution/src/aggregate_shard_reports.py --data-root '..\student_resource\dataset' --reports D:\temp\train-0-report.json D:\temp\train-1-report.json D:\temp\train-2-report.json D:\temp\train-3-report.json --target-count 10320219 --cap 64 --logs D:\temp\train-0.log D:\temp\train-1.log D:\temp\train-2.log D:\temp\train-3.log --out D:\temp\person-a-full-train-report.json
```

The aggregator checks that each report's Source 1 and true-link counts match
its disjoint shard before combining recall, complete-set coverage, oracle
macro F0.5, candidate volume, reduction ratio, and resource telemetry. It
reports global p95 when every shard has p95 equal to the common hard cap;
otherwise full candidate-count histograms are needed for an exact global p95.

The retrieval source contains **experimental** token channels. `cross_token`
intersects compact name-token and address-token postings with a
1,000-document-frequency probe ceiling. `name_pair` and `address_pair` propose
records sharing two informative words within one field, including cases where
the other field is absent or entirely different. `RETRIEVAL_VERSION` separates
these checkpoints from the older baseline and cross-token-only version. The
private `amazon-ml-person-a-dev-cross-token` and
`amazon-ml-person-a-dev-word-pair` jobs use the same seeded 10k development
Source 1 IDs, full train targets, and cap 64; their outputs must be measured
before selecting either for full train/test generation. Do not combine staged
parts across retrieval versions.

The completed baseline `dev` job was run before `cross_token` existed. Its
cap-64 reassembly is stored outside Git at `D:\temp\amazon-ml-dev-cap64.tsv`.
The current `assemble_staged.py` intentionally rejects that older work
directory because its retrieval version differs. For any new development run,
download that run's staged parts and reassemble lower caps with the same code
version that produced them.

## Private Kaggle CPU execution

The private official dataset is
`mridulnegi2005/amazon-ml-challenge-2026-private-input` and the private code
dataset is `mridulnegi2005/amazon-ml-person-a-code`. The Kaggle CLI is used to
upload/run/download; keep both datasets and every kernel private.

```powershell
python docs/person-a/package_kaggle_code.py --out D:\temp\amazon-ml-code-upload
kaggle datasets version -p D:\temp\amazon-ml-code-upload -m 'Describe source change'
```

Create a private Kaggle **script** kernel with `kaggle_launcher.py` as its
`code_file`; attach both dataset slugs in `dataset_sources`, set
`enable_gpu: "false"` and `enable_internet: "true"` (for pip installation).
Use a unique title and matching ID per mode; set `MODE` in the launcher to
`smoke`, `dev`, `holdout`, `train`, `test`, or `test-0-of-4` (similarly for
the other shards). `NAME_K` and `ADDRESS_K` in the launcher default to 32 and
16; record any experimental changes in the corresponding kernel's title and
run log. The launcher locates attached
files under `/kaggle/input`, runs the candidate CLI, and writes output under
`/kaggle/working`. Use the CLI to operate each kernel:

```powershell
kaggle kernels push -p D:\temp\amazon-ml-kernel-dev
kaggle kernels status mridulnegi2005/amazon-ml-person-a-dev-10k
kaggle kernels output mridulnegi2005/amazon-ml-person-a-dev-10k -p D:\temp\amazon-ml-kernel-dev-output
```

Do not treat `smoke` output as a quality estimate: it uses only the first
1,000 Source 1 rows and one million targets per source. `dev` and `holdout`
use the full target sources and can report true-link recall, complete-set
coverage, candidate volume, and oracle macro F0.5. Oracle F0.5 is the best
possible matching score if B classifies every retrieved pair perfectly, not
the actual team score. The `test` mode has no ground truth and never reports a
score.
