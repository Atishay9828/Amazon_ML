# Business entity resolution: candidate generation

This package finds, for each Source 1 business record, a short list of Source 2 and Source 3 records that can describe the same business. The matching model scores only this list. A true match that is not in the list cannot be found later. The package therefore optimizes candidate recall at a fixed list size.

The package uses only the challenge data. It uses no external business data, geocoding, or lookup service.

## What the pipeline does

The pipeline has four steps. One configuration file controls all steps.

1. **Reverse search.** Each Source 2 and Source 3 record finds its most similar Source 1 record. The search uses character 3-grams of the name and the address. A second pass uses the name alone for records that have no address.
2. **Forward retrieval.** Each Source 1 record retrieves up to 256 records from each target source. The retrieval uses these channels:
   - exact normalized name and address;
   - rare words, word pairs, and cross-field word matches;
   - character 3–5 grams, selected by a stable hash;
   - a second hop from strong first candidates to their close neighbours;
   - exact keys that survive Indian-script transliteration, digit-letter confusion, joined website names, and street numbers;
   - the pairs from step 1.
3. **Ranker training.** A gradient-boosted classifier learns which retrieved records are true matches. It trains on 50,000 training Source 1 records. These records are not in the development or holdout samples.
4. **Final selection.** The ranker scores every retrieved record. The pipeline keeps the 32 best records for each Source 1 record, 16 from each source first.

Country is a ranker feature only. The pipeline never filters by country, so France records in the test data are processed like other records.

## Configuration

- `configs/final.json` holds every value for every step. The fields that start with `_` are notes. They do not change the result.
- `configs/baseline.json` holds the historical defaults. Use it only to reproduce old experiments.
- Each output records a SHA-256 hash of the configuration. The pipeline refuses to mix staged files, reverse files, or ranker models from different configurations.

## Measured recall

The development sample has 10,000 training Source 1 records and 34,770 true links. It searches all 10,320,219 training targets.

| Configuration | Candidate recall at 32 candidates |
| --- | ---: |
| Hand-weighted score, before this pipeline | 90.79% |
| Broad channels with learned ranker | 95.44% |
| Plus second hop, hash grams, and reverse search | 96.04% |
| Plus name keys | 96.44% |

These values are candidate recall. They are not the matcher score or the leaderboard score.

## Output

The pipeline writes a sorted tab-separated file with one row for each candidate pair:

```text
source1_entity_id	candidate_entity_id	name_cosine	address_cosine	retrieval_channels
```

A Source 1 record without candidates has no row. The wide `candidate_pairs.tsv` of the submission must contain one row for every test Source 1 record, with an empty list where necessary.

## Run the pipeline

1. Install Python 3.12 and the dependencies:

   ```bash
   python -m pip install -r code/business_entity_resolution/requirements.txt
   ```

2. Run the tests:

   ```bash
   python -m unittest discover -s code/business_entity_resolution/tests
   ```

3. Check the configuration on the development sample:

   ```bash
   python code/business_entity_resolution/src/run_pipeline.py --config code/business_entity_resolution/configs/final.json --data-root dataset --split train --sample-split dev --work-dir work --out work/dev.tsv
   ```

4. Make the test candidate set:

   ```bash
   python code/business_entity_resolution/src/run_pipeline.py --config code/business_entity_resolution/configs/final.json --data-root dataset --split test --work-dir work --out output/candidates-long.tsv
   ```

The `--data-root` folder must contain `train/` and `test/` with the official files. The pipeline resumes from matching checkpoints in `--work-dir`. For a split into shards, add `--shard-index` and `--shard-count`, then merge with `--merge-shards`. A full run needs a Linux machine with about 30 GB of memory and 4 CPU cores. See `docs/person-a/runbook.md` for details.
