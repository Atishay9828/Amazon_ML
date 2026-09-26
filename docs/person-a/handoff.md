# Person A handoff — 26 September 2026, 16:48 IST

Owner: Mridul. Branch: `feat/data-candidates`. Person B owns matching and
evaluation; Atishay owns integration, the final zip, and portal uploads.

## Candidate interface

Run `code/business_entity_resolution/src/candidates.py` with `--data-root`,
`--split train|test`, and `--out`. It emits a sorted, unique long-form TSV with
exactly these columns:

```text
source1_entity_id	candidate_entity_id	name_cosine	address_cosine	retrieval_channels
```

Each row is one Source 1 to Source 2/3 pair. Source 1 entities with no
candidates have no long-form row. The final `candidate_pairs.tsv` must be
converted to the challenge's wide format with **one row for every test Source 1
entity**, including an empty list where needed. The exact long-form pair set
given to B's scorer must match that wide candidate set; every final predicted
match must be in it. See `runbook.md` for validation and Kaggle commands.

## Measured fallback available now

The bounded broad-token configuration uses `--name-k 32 --address-k 16
--indexed-grams 16 --query-grams 32 --pair-max-df 5000 --pair-max-hits 2000
--single-max-df 512 --single-tokens 2 --stage-cap 64 --cap 64
--final-score balanced`. It uses no external business lookup and no hard
country filter.

On the seeded 10,000 Source 1 development IDs, balanced final selection over
its saved stage kept **32,840/34,770 true links (94.4492% recall)** in 639,042
pairs, with 0.980054 oracle macro F0.5. On the separate fixed 50,000-ID
holdout, the same configuration kept **162,720/172,581 links (94.2862%)** in
3,194,567 pairs, with 0.977965 oracle macro F0.5. The holdout run finished in
1,666.20 seconds and peaked at 21.984 GiB container memory. These are
candidate-generation metrics, **not** B's matcher or leaderboard score.

The full-training and full-test tables for this fallback are **not complete**.
Full-training shards 0 and 1 of 4 are running privately on Kaggle as
`mridulnegi2005/amazon-ml-person-a-train-broad-balanced-{0,1}-of-4`. An older
word-pair configuration completed all labeled training at 90.6929% recall,
but those shards do not measure the newer fallback. A validated older
full-test long-form table exists outside Git at
`D:\temp\amazon-ml-test-cap64.tsv` (110,336,665 pairs); it is a fallback
artifact, not the broad-token/balanced output. One test Source 1 ID has zero
candidates in that older table, so its wide-format row must be empty.

## Newer proposal experiments and active runs

The isolated name-key job added 211 true links to the staged 128-row candidate
set but lost 78 final links against the controlled broad-token stage128. Wider
word probes kept 33,026/34,770 true development links at cap64; 32/64
character grams kept 32,975. Same-source second-hop search with two seeds and
a 2,048-row stage kept **33,139/34,770 (95.3092%)** at final cap64, in
639,876 pairs, with 0.982735 oracle macro F0.5. Reconstructing a practical
128-row stage kept **33,125/34,770 (95.2689%)** and 0.982615 oracle; this is
the strongest current development setting. Its complete event took 1,507.67
seconds and peak container memory was 21.846 GiB. These are candidate oracle
metrics, not B's matcher score.

The private `amazon-ml-person-a-holdout-sibling128` kernel and full-test
`amazon-ml-person-a-test-sibling128-{0,1}-of-4` kernels are running with the
actual 128-row sibling setting. Their results and final cap decision are
pending. Full-training fallback shards 0/1 remain running under the older
broad-token/balanced setting; do not mix their outputs with sibling outputs.
At cap96, the sibling development stage kept 33,284/34,770 true links in
958,624 pairs with 0.984159 oracle macro F0.5; cap128 kept 33,360 links in
1,268,043 pairs with 0.984841 oracle; cap256 kept 33,462 links in 2,288,941
pairs with 0.985960 oracle. Cap96 is provisionally the smallest tested set
within 0.002 oracle F0.5 of the best tested cap256 result. The `docs/data_audit.md` experiment
ledger and `docs/person-a/todo.md` run log are the authoritative result records.

## Reproduction and remaining gate

- A-owned source and tests are under `code/business_entity_resolution/`; pinned
  dependencies are in `docs/person-a/requirements.txt`.
- Run `python -m unittest discover -s code/business_entity_resolution/tests -q`
  from the repository root; 19 tests passed on 26 September.
- Private Kaggle data and code dataset slugs, launcher setup, report
  aggregation, stage diagnostics, and long-form validation commands are in
  `docs/person-a/runbook.md`.
- Before calling A's handoff final, choose the configuration, generate **both
  full train and full test** tables with identical settings, validate the
  long-form output, record measured resource use, and give B/C the artifact
  paths and exact run commands. C then converts and validates the wide
  `candidate_pairs.tsv` and packages the submission.
