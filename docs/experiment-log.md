# Experiment Log: Business Entity Resolution (Amazon ML Challenge 2026)

This document records every experiment of the pipeline, with its setting, its result, and the decision that followed. All recall and F0.5 values come from train records with known answers. Leaderboard values come from the challenge portal.

## 1. Metrics used

| Metric | Meaning |
|---|---|
| Macro F0.5 | The challenge metric. F0.5 per Source 1 record, averaged. A record with no true match scores 1.0 only with an empty prediction. |
| Candidate recall | The share of true links that are in the candidate set. |
| Oracle F0.5 | The macro F0.5 of a perfect matcher on the candidate set. It is the ceiling for that candidate set. |
| CV | Grouped 5-fold cross-validation by Source 1 record. Each record is predicted by a model that did not train on it. |

Evaluation sets:

| Set | Size | Use |
|---|---|---|
| dev | 10,000 train Source 1 records (seeded) | Retrieval experiments, first matcher experiments |
| holdout | 50,000 train Source 1 records | Matcher training and CV |
| ranktrain | 50,000 train Source 1 records, disjoint from dev and holdout | Ranker training; later extra matcher training data |

## 2. Data facts

- Train: 2,206,821 Source 1 records and 10,320,219 Source 2 and 3 records. Test: 1,732,544 Source 1 records and 9,969,589 Source 2 and 3 records.
- A Source 1 record has 3.5 true matches on average. 5.6% of records have no match.
- Every Source 2 or 3 record matches at most one Source 1 record (7,638,365 matched train targets, no exceptions). About 26% of targets match no record.
- The test set adds France: 259,452 Source 1 records (15%). Train has no French records.
- Leakage checks found no signal: true pairs and random pairs have the same ID correlation (0.0005 and -0.0016), the same row-position correlation (0.0055 and -0.0029), and the same last-3-digit agreement (0.09% and 0.10%). No test ID occurs in train.

## 3. Retrieval (blocking) experiments

All results: dev 10,000 records, 34,770 true links.

### 3.1 Configurations

| Run | Setting | Learned cap 32 recall | Other |
|---|---|---|---|
| R4 | Broad token channels and char n-gram TF-IDF (proposal ceiling) | 95.67% | — |
| C1 | R4 + reverse top-1 (DF cap 0.2%) | 96.04% | staged 96.33% |
| C2 | C1 + name keys | 96.44% | name keys displaced true links under the old rank rule |
| C3 | C1 with 4 hop seeds, K 32 | 95.96% | hop 2/16 kept |
| C4 | C1 without hashed gram selection | 95.93% | hashed selection kept |
| **C5** | C1 + four exact keys (phonetic, OCR, compact, numbers), key cap 200 | **97.44%** | cap 64: 97.56%, staged 98.17%, oracle 0.9918 |

### 3.2 Reverse search DF cap

| DF cap | Top-1 recall of reverse search | New true links over C5 | ms per query | Train pass (4 workers) |
|---|---|---|---|---|
| 0.2% | 50.9% | +47 | 1.46 | 1.05 h |
| 0.3% | 64.0% | +81 | 2.36 | 1.69 h |
| 0.5% | 78.3% | +132 | 3.89 | 2.79 h |
| 0.75% | 85.2% | +183 | 12.66 | 9.07 h |
| 1% | 88.5% | +219 | 16.13 | 11.56 h |

Decision: test reverse at 1% (7.1 h on Kaggle), train reverse at 0.5% (2.7 h). The 1% train reverse did not finish inside the 12-hour Kaggle limit.

### 3.3 Final retrieval configuration (`configs/final.json`)

- char 3-5 gram TF-IDF (2^20 hashed features) on name and on address, hashed gram selection, second hop (2 seeds, K 16);
- reverse top-1 (DF cap 1% on test, 0.5% on train) and blank-address reverse search (top 20);
- four exact keys with key cap 200;
- stage cap 256 per source; learned ranker final cap 32 (16 per source).

Measured on the final pipeline (dev 10,000): staged recall **98.37%**, rule-based control cap 32 recall 90.45%.

## 4. Learned ranker

### 4.1 Defect found

The repository ranker (`rank_candidates.train`) kept all positives but only the 32 negatives per source with the highest base rank. Low-base-rank rows therefore appeared in training only as positives. At inference the model ranked about 200 low-base-rank negatives per record above true matches.

| Ranker | Cap 32 recall (dev) | Oracle F0.5 |
|---|---|---|
| Repository ranker | **53.9%** | 0.757 |
| Rule-based control (no model) | 90.5% | 0.956 |
| Retrained with hard negatives + 64 weighted random negatives per source (half-dev check) | 97.2% | 0.991 |
| Same, trained on full dev, holdout 5k check | **97.07%** | 0.990 |
| Same, 300 iterations without early stopping | 96.34% | 0.988 |

Decision: the early-stopped retrained ranker (scikit-learn 1.6.1, the Kaggle version) ranks all test shards.

### 4.2 Other ranker defect

The ranker signature compared the blank-address pair file digest. Train and test digests always differ, so every test inference failed the signature check. Commit `2a10a8a` removes the digest from the signature.

### 4.3 Cap 32 versus cap 64 (holdout 50,000)

| Cap | Candidate recall | Oracle F0.5 | Matcher CV |
|---|---|---|---|
| 32 | 96.97% | 0.9892 | **0.9661** |
| 64 | 97.10% | 0.9897 | 0.9654 |

Decision: keep cap 32.

## 5. Matcher experiments

### 5.1 Features (CV on dev 10,000, C5 candidates, oracle 0.9918)

| Step | Stage 1 | Stacked | Decision |
|---|---|---|---|
| Base features (rapidfuzz ratios, Jaccard, keys, channels, group rank) | 0.9288 | 0.9298 | start |
| + noise-tolerant name, house-number closeness, group address structure | 0.9445 | 0.9458 | kept |
| + word alignment (extra, missing, substituted words) | 0.9474 | 0.9476 | kept |
| + name-word rarity (split-wide IDF) | 0.9561 | 0.9562 | kept |
| + whole-name frequency | 0.9595 | 0.9599 | kept |
| + address-key frequency | 0.9598 | **0.9623** | kept |
| + address-word rarity | 0.9552 | 0.9556 | rejected |
| + key frequency (name+address, house+street) | 0.9607 | 0.9626 | rejected (noise, 1.9 GB tables) |
| + phonetic word skeletons | 0.9615 | 0.9618 | rejected on 10k |
| Larger LightGBM (1,500 rounds, 127 leaves) | 0.9470 | 0.9464 | rejected |
| Smaller LightGBM (31 leaves) | 0.9598 | 0.9613 | rejected |
| Third stacking stage | — | 0.9617 | rejected |
| Seed averaging (3 seeds) | 0.9606 | 0.9618 | rejected |
| Decision-rule changes (thresholds, powers, empty bias) | — | at most +0.0005 | rejected |
| Post-rules (add exact name and address duplicates) | — | 0.9465 to 0.9475 | rejected |

### 5.2 Training data (final pipeline candidates)

| Training records | CV on the 50,000 holdout records |
|---|---|
| 50,000 (holdout only) | 0.9661 (stage 1 0.9621) |
| 100,000 (holdout + ranktrain) | **0.9675** (0.9670 over all 100,000) |

### 5.3 Matcher variants on 100,000 records

| Variant | Stacked CV |
|---|---|
| Base | 0.96705 |
| Bigger model (1,500 rounds, 127 leaves) | 0.96769 |
| Wider trees (255 leaves) | 0.96748 |
| Phonetic features | 0.96758 |
| Phonetic + bigger model | 0.96829 |

The cross-encoder blend uses the base matcher, because the blend was fitted on its probabilities.

### 5.4 Probability calibration

Each probability bin is within 0.015 of the observed match rate. Recall-leaning decisions reduce CV (power 0.9: 0.96678; power 0.8: 0.96635).

### 5.5 Loss breakdown (100,000 records, loss 3.3 points)

| Group | Records | Loss (points) |
|---|---|---|
| India, cross-script | 12,318 | 0.79 |
| India, Latin | 25,430 | 0.83 |
| US, Latin | 56,649 | 1.35 |
| No-match records | 5,603 | 0.32 |

Missed links (29,787) are ten times more frequent than wrong links (2,790).

## 6. Target exclusivity

Rule: each Source 2 or 3 record goes to the Source 1 record with the highest probability. The other records re-decide without it.

- Holdout effect: 0.96611 to 0.96614 (the sample holds few competing records).
- Test effect (A2 to B2): 19,044 targets claimed by several records became 6 (ties); 16,048 records changed.

## 7. Cross-encoders (GPU)

| Setting | Value |
|---|---|
| Pairs | 170,767 uncertain train pairs (matcher probability 0.01 to 0.99), 45,885 true |
| Folds | 2-fold by Source 1 for out-of-fold scores; one model on all pairs for test |
| Blend | LightGBM on matcher probability, cross-encoder score, and group ranks; 5-fold CV |

| Model | AUC on uncertain pairs | Blend CV (100,000) |
|---|---|---|
| Matcher alone | 0.9546 | 0.96704 |
| `mmarco-mMiniLMv2-L12-H384` (P100 run) | 0.9607 | **0.97557** |
| Same model, second run (T4); average of both runs | 0.9608; 0.9613 | 0.97548 |
| `xlm-roberta-base` | 0.9649 | 0.97571 |
| Both models as separate features | — | **0.97628** |

Uncertain test pairs scored: 3,802,479.

## 8. Dense retrieval (GPU)

- Embeddings: multilingual-e5-small, "name, address", int8 storage for test.
- Search: GPU cosine top-20 (train) and top-10 (test) over all targets.

Recall gain on 100,000 train records (base candidate recall 96.94%):

| Dense top-k | New pairs | New true links | Recall gain | Share of new pairs that are true |
|---|---|---|---|---|
| 1 | 1,376 | 422 | +0.12 points | 30.7% |
| 3 | 21,089 | 2,215 | +0.64 points | 10.5% |
| 5 | 97,367 | 3,949 | +1.14 points | 4.1% |
| 10 | 479,967 | 5,228 | +1.51 points | 1.1% |
| 20 | 1,384,600 | 5,809 | +1.68 points | 0.4% |

Scoring of the top-5 new pairs: small cross-encoder, then a 5-fold logistic calibration on (cross-encoder logit, cosine, log rank). AUC 0.9995.

| Setting | CV (100,000) |
|---|---|
| B5 path (matcher + two cross-encoders) | 0.97628 |
| + dense top-5 pairs | **0.98044** |

The cross-encoder that scored the train pairs trained on other pairs of the same records. This CV can therefore be optimistic.

## 9. Other experiments

| Experiment | Result | Decision |
|---|---|---|
| e5 embedding cosine as matcher features | +0.0003 CV | rejected |
| France duplicate-name hypothesis | French duplicate-name share 34% (US 29-36%, India 45%) | rejected |
| Early fast matcher on candidate-file columns only | dev 0.790 | fallback only |
| Matcher trained on older-pipeline candidates | 0.85 to 0.87 | rejected: training candidates must come from the final pipeline |

## 10. Submissions

| Submission | Content | CV | Public leaderboard |
|---|---|---|---|
| Fallback | Column-only matcher on baseline candidates | 0.790 (dev) | not submitted |
| A / B | Matcher (50,000), plain / exclusive | 0.9661 | — |
| A2 / B2 | Matcher (100,000), plain / exclusive | 0.9675 | — |
| B3 | + small cross-encoder blend, exclusive | 0.9756 | **0.9685** (rank 1,680) |
| B4 | + large cross-encoder instead | 0.9757 | — |
| B5 | + both cross-encoders, exclusive | 0.9763 | **0.970** |
| B6 | + dense top-5 channel, exclusive | 0.9804 | pending |

Every submitted file passed `utils/validate_submission.py`, and every match is inside `candidate_pairs.tsv`.

## 11. Compute

- Kaggle: two accounts, 5 CPU sessions each (4 cores, 30 GB RAM, 12-hour limit), 2 GPU sessions (P100 or T4 x2), 30 GPU hours per week.
- Jobs killed at the 12-hour limit: 1% train reverse holdout job, P1 dev check, V2 name keys, dense D1 on CPU.
- T4 queues delayed several GPU jobs by up to 45 minutes. P100 requests started at once.
- Local laptop: 8 logical cores, 16 GB RAM. Full-file validation needs about 8 GB.

## 12. Lessons

1. Validate each new pipeline component end to end on labelled data before test inference. The ranker defect cost one full test inference round.
2. The largest single gains came from GPU models (cross-encoders +0.85, dense retrieval +0.42 CV). They should have started earlier.
3. Benchmark cost before a long run. The 1% train reverse needed more than 12 hours.
4. Save intermediate outputs to the job output folder. One holdout file was written to a temporary folder and had to be rebuilt.
5. Local CV and the leaderboard differ by about 0.7 points. France (15% of test, no labels) is the likely cause.
