# Person A candidate generation runbook

This is the long-form candidate table handed to Person B for scoring. The
official wide-format `candidate_pairs.tsv` and `matching_results.tsv` are
assembled by Person C. No official data, candidate output, or credentials
belong in Git.

## Local commands

From the repository root, with Python 3.12+ and the A-owned pinned
dependencies in `docs/person-a/requirements.txt`:

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
`smoke`, `dev`, `holdout`, `train`, or `test`. The launcher locates attached
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
