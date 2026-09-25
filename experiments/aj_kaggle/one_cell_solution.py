"""AJ's self-contained Kaggle experiment. The notebook contains this whole file as one code cell.

Official challenge data stays outside Git. This is an experimental, measurable
pipeline, not a claim of a winning or Kaggle-verified score.
"""

import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import xgboost as xgb


SEED = 2026
QUERY_SAMPLE_MOD = 15  # 5% validation, 10% training; all target rows remain searchable.
VALID_MOD = 5
MAX_TOKEN_DF = 64
MAX_EXACT_DF = 512
MAX_TOKEN_KEYS_PER_CHANNEL = 3
MAX_CANDIDATES_PER_SOURCE = 40
CANDIDATE_VERSION = "v3"
DUCKDB_MEMORY = "9GB"
THREADS = max(2, min(8, os.cpu_count() or 4))
FEATURES = [
    "name_jw", "address_jw", "name_jaccard", "address_jaccard",
    "name_length_ratio", "address_length_ratio", "number_equal",
    "number_both_present", "exact_name", "exact_address",
    "key_hits", "name_key", "address_key", "number_address_key",
]


def locate_dataset():
    roots = [Path("/kaggle/input"), Path.cwd()]
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("**/dataset/train/train_ground_truth.tsv"))
    roots_found = sorted({p.parent.parent.resolve() for p in candidates})
    if len(roots_found) != 1:
        raise RuntimeError(
            f"Expected one official dataset under Kaggle input or cwd; found {roots_found}. "
            "Set DATA_ROOT explicitly if you attached more than one copy."
        )
    return roots_found[0]


DATA_ROOT = locate_dataset()
WORK = Path("/kaggle/working/aj_entity_resolution") if Path("/kaggle/working").exists() else Path.cwd() / "work" / "aj_entity_resolution"
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "tmp").mkdir(exist_ok=True)
(WORK / "models").mkdir(exist_ok=True)
(WORK / "train_candidates").mkdir(exist_ok=True)
(WORK / "test_candidates").mkdir(exist_ok=True)
(WORK / "accepted").mkdir(exist_ok=True)
OUTPUT = WORK / "output"
OUTPUT.mkdir(exist_ok=True)


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


if torch.cuda.device_count() < 2:
    raise RuntimeError("This run requires Kaggle's 2 x T4 GPU accelerator. Enable it and restart.")
log(f"Data: {DATA_ROOT}; GPUs: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_name(1)}")
log(f"Versions: DuckDB {duckdb.__version__}, XGBoost {xgb.__version__}, Torch {torch.__version__}")


def sq(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha256_stream(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def con(memory=DUCKDB_MEMORY):
    db = duckdb.connect()
    db.execute(f"SET memory_limit = {sq(memory)}")
    db.execute(f"SET threads = {THREADS}")
    db.execute(f"SET temp_directory = {sq(WORK / 'tmp')}")
    db.execute("SET preserve_insertion_order = false")
    return db


def csv_source(path):
    return f"read_csv({sq(path)}, delim='\\t', header=true, all_varchar=true)"


def normalized_record_sql(path, country, is_query, sampled):
    # Generic accent folding and punctuation treatment; no country whitelist.
    filt = f"country = {sq(country)}"
    if is_query and sampled:
        filt += f" AND hash(entity_id) % 100 < {QUERY_SAMPLE_MOD}"
    src = csv_source(path)
    return f"""
        SELECT entity_id AS rid,
          trim(regexp_replace(regexp_replace(
            ' ' || trim(regexp_replace(lower(strip_accents(coalesce(business_name,''))), '[^a-z0-9]+', ' ', 'g')) || ' ',
            ' (incorporated|inc|corporation|corp|limited|ltd|llc|llp|private|pvt) ', ' ', 'g'),
            ' +', ' ', 'g')) AS nm,
          trim(regexp_replace(lower(strip_accents(coalesce(business_address,''))), '[^a-z0-9]+', ' ', 'g')) AS ad,
          regexp_extract(coalesce(business_address,''), '[0-9]+') AS num
        FROM {src}
        WHERE {filt}
    """


def keys_sql(table):
    # Separate exact, name-token, address-token, and house-number/address-token
    # channels. The document-frequency gate below prevents giant Cartesian blocks.
    return f"""
      SELECT DISTINCT rid, key FROM (
        SELECT rid, 'N=' || nm AS key FROM {table} WHERE length(nm) >= 3
        UNION ALL
        SELECT rid, 'A=' || ad AS key FROM {table} WHERE length(ad) >= 5
        UNION ALL
        SELECT rid, 'NT=' || tok AS key
          FROM {table}, UNNEST(str_split(nm, ' ')) AS u(tok)
          WHERE length(tok) >= 3 AND regexp_matches(tok, '[a-z]')
        UNION ALL
        SELECT rid, 'AT=' || tok AS key
          FROM {table}, UNNEST(str_split(ad, ' ')) AS u(tok)
          WHERE length(tok) >= 4 AND regexp_matches(tok, '[a-z]')
        UNION ALL
        SELECT rid, 'AN=' || num || ':' || tok AS key
          FROM {table}, UNNEST(str_split(ad, ' ')) AS u(tok)
          WHERE length(num) >= 2 AND length(tok) >= 4 AND regexp_matches(tok, '[a-z]')
      )
    """


def build_partition(split, country, source):
    folder = WORK / f"{split}_candidates"
    safe_country = hashlib.sha1(country.encode()).hexdigest()[:10]
    out = folder / f"{source}_{safe_country}_{CANDIDATE_VERSION}_{MAX_TOKEN_DF}_{MAX_EXACT_DF}_{MAX_TOKEN_KEYS_PER_CHANNEL}_{MAX_CANDIDATES_PER_SOURCE}.parquet"
    if out.exists():
        log(f"Reuse candidate partition: {out.name}")
        return out
    qpath = DATA_ROOT / split / f"{split}_source1.tsv"
    tpath = DATA_ROOT / split / f"{split}_{source}.tsv"
    db = con()
    try:
        log(f"Blocking {split} {country} {source}")
        db.execute(f"CREATE TEMP TABLE q AS {normalized_record_sql(qpath, country, True, split == 'train')}")
        db.execute(f"CREATE TEMP TABLE t AS {normalized_record_sql(tpath, country, False, False)}")
        nq, nt = db.execute("SELECT (SELECT count(*) FROM q), (SELECT count(*) FROM t)").fetchone()
        log(f"  queries={nq:,}, targets={nt:,}")
        if not nq or not nt:
            db.execute(f"COPY (SELECT ''::VARCHAR qid, ''::VARCHAR tid, {', '.join(f'0::FLOAT {x}' for x in FEATURES)} WHERE false) TO {sq(out)} (FORMAT parquet, COMPRESSION zstd)")
            return out
        db.execute(f"CREATE TEMP TABLE tk AS {keys_sql('t')}")
        db.execute(f"CREATE TEMP TABLE qk AS {keys_sql('q')}")
        db.execute(f"""
            CREATE TEMP TABLE usable AS
            SELECT key, count(*) AS freq FROM tk GROUP BY key
            HAVING count(*) <= CASE WHEN starts_with(key, 'N=') OR starts_with(key, 'A=')
                THEN {MAX_EXACT_DF} ELSE {MAX_TOKEN_DF} END
        """)
        db.execute(f"""
            CREATE TEMP TABLE qku AS
            SELECT qk.rid, qk.key FROM qk JOIN usable USING (key)
            QUALIFY row_number() OVER (
              PARTITION BY qk.rid,
                CASE WHEN starts_with(qk.key, 'N=') THEN 'N='
                     WHEN starts_with(qk.key, 'A=') THEN 'A='
                     WHEN starts_with(qk.key, 'NT=') THEN 'NT='
                     WHEN starts_with(qk.key, 'AT=') THEN 'AT='
                     ELSE 'AN=' END
              ORDER BY usable.freq, qk.key
            ) <= CASE WHEN starts_with(qk.key, 'N=') OR starts_with(qk.key, 'A=')
                 THEN 1 ELSE {MAX_TOKEN_KEYS_PER_CHANNEL} END
        """)
        db.execute("""
            CREATE TEMP TABLE pairs AS
            SELECT qku.rid AS qid, tk.rid AS tid,
              count(*)::FLOAT AS key_hits,
              max(CASE WHEN starts_with(qku.key, 'N=') OR starts_with(qku.key, 'NT=') THEN 1 ELSE 0 END)::FLOAT AS name_key,
              max(CASE WHEN starts_with(qku.key, 'A=') OR starts_with(qku.key, 'AT=') THEN 1 ELSE 0 END)::FLOAT AS address_key,
              max(CASE WHEN starts_with(qku.key, 'AN=') THEN 1 ELSE 0 END)::FLOAT AS number_address_key
            FROM qku JOIN tk USING (key)
            GROUP BY qku.rid, tk.rid
        """)
        pair_count = db.execute("SELECT count(*) FROM pairs").fetchone()[0]
        log(f"  raw candidate pairs={pair_count:,}")
        feature_query = f"""
          WITH f AS (
            SELECT p.qid, p.tid, p.key_hits, p.name_key, p.address_key, p.number_address_key,
              jaro_winkler_similarity(q.nm, t.nm)::FLOAT AS name_jw,
              CASE WHEN q.ad = '' OR t.ad = '' THEN 0 ELSE jaro_winkler_similarity(q.ad, t.ad) END::FLOAT AS address_jw,
              CASE WHEN q.nm = '' OR t.nm = '' THEN 0
                ELSE jaccard(q.nm, t.nm) END::FLOAT AS name_jaccard,
              CASE WHEN q.ad = '' OR t.ad = '' THEN 0 ELSE jaccard(q.ad, t.ad) END::FLOAT AS address_jaccard,
              (least(length(q.nm), length(t.nm))::FLOAT / greatest(1, length(q.nm), length(t.nm)))::FLOAT AS name_length_ratio,
              (least(length(q.ad), length(t.ad))::FLOAT / greatest(1, length(q.ad), length(t.ad)))::FLOAT AS address_length_ratio,
              CASE WHEN q.num <> '' AND q.num = t.num THEN 1 ELSE 0 END::FLOAT AS number_equal,
              CASE WHEN q.num <> '' AND t.num <> '' THEN 1 ELSE 0 END::FLOAT AS number_both_present,
              CASE WHEN q.nm <> '' AND q.nm = t.nm THEN 1 ELSE 0 END::FLOAT AS exact_name,
              CASE WHEN q.ad <> '' AND q.ad = t.ad THEN 1 ELSE 0 END::FLOAT AS exact_address
            FROM pairs p JOIN q ON p.qid = q.rid JOIN t ON p.tid = t.rid
          )
          SELECT qid, tid, {', '.join(FEATURES)} FROM f
          QUALIFY row_number() OVER (
            PARTITION BY qid ORDER BY
              0.38 * greatest(name_jw, address_jw) + 0.28 * name_jw + 0.20 * address_jw + 0.07 * number_equal +
              0.05 * exact_name + 0.04 * least(key_hits, 3) / 3 DESC, tid
          ) <= {MAX_CANDIDATES_PER_SOURCE}
        """
        db.execute(f"COPY ({feature_query}) TO {sq(out)} (FORMAT parquet, COMPRESSION zstd)")
        kept = db.execute(f"SELECT count(*) FROM read_parquet({sq(out)})").fetchone()[0]
        log(f"  model-input pairs={kept:,} (cap {MAX_CANDIDATES_PER_SOURCE} per source/query)")
        return out
    finally:
        db.close()
        gc.collect()


def country_labels(split):
    db = con()
    try:
        path = DATA_ROOT / split / f"{split}_source1.tsv"
        return [r[0] for r in db.execute(f"SELECT DISTINCT country FROM {csv_source(path)} ORDER BY country").fetchall()]
    finally:
        db.close()


train_countries = country_labels("train")
test_countries = country_labels("test")
log(f"Country labels: train={train_countries}, test={test_countries}")
train_parts = []
test_parts = []
for country in train_countries:
    for source in ("source2", "source3"):
        train_parts.append(build_partition("train", country, source))
for country in test_countries:
    for source in ("source2", "source3"):
        test_parts.append(build_partition("test", country, source))


def parquet_files(parts):
    return "[" + ",".join(sq(p) for p in parts) + "]"


def make_labels():
    out = WORK / f"sampled_positive_links_mod{QUERY_SAMPLE_MOD}.parquet"
    if out.exists():
        return out
    db = con()
    try:
        gt = csv_source(DATA_ROOT / "train" / "train_ground_truth.tsv")
        db.execute(f"""
          COPY (
            SELECT source1_entity_id AS qid, unnest(str_split(matched_entity_ids, ',')) AS tid
            FROM {gt}
            WHERE hash(source1_entity_id) % 100 < {QUERY_SAMPLE_MOD}
              AND matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
          ) TO {sq(out)} (FORMAT parquet, COMPRESSION zstd)
        """)
        return out
    finally:
        db.close()


LABELS = make_labels()


def train_one(source, gpu):
    parts = [p for p in train_parts if p.name.startswith(source + "_")]
    db = con("5GB")
    try:
        files = parquet_files(parts)
        labels = sq(LABELS)
        cols = ", ".join("f." + x for x in FEATURES)
        common = f"""
          SELECT f.qid, f.tid, {cols}, CAST(l.tid IS NOT NULL AS UTINYINT) AS y
          FROM read_parquet({files}) f
          LEFT JOIN read_parquet({labels}) l ON f.qid = l.qid AND f.tid = l.tid
        """
        dev = db.execute(common + f" WHERE hash(f.qid) % 100 >= {VALID_MOD}").df()
        val = db.execute(common + f" WHERE hash(f.qid) % 100 < {VALID_MOD}").df()
        if dev.empty or val.empty or dev.y.nunique() < 2:
            raise RuntimeError(f"No usable training/validation rows for {source}")
        log(f"{source} GPU {gpu}: dev={len(dev):,}, val={len(val):,}, dev positives={int(dev.y.sum()):,}")
        train_x = np.ascontiguousarray(dev[FEATURES].to_numpy(dtype=np.float32))
        val_x = np.ascontiguousarray(val[FEATURES].to_numpy(dtype=np.float32))
        dtrain = xgb.DMatrix(train_x, label=dev.y.to_numpy(dtype=np.uint8), feature_names=FEATURES)
        dval = xgb.DMatrix(val_x, label=val.y.to_numpy(dtype=np.uint8), feature_names=FEATURES)
        del dev, train_x, val_x
        gc.collect()
        params = {
            "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
            "device": f"cuda:{gpu}", "max_depth": 8, "eta": 0.07,
            "max_bin": 256, "subsample": 0.85, "colsample_bytree": 0.9,
            "min_child_weight": 8, "lambda": 4.0, "seed": SEED + gpu,
            "nthread": max(2, THREADS // 2),
        }
        booster = xgb.train(params, dtrain, num_boost_round=750,
                            evals=[(dval, "valid")], early_stopping_rounds=50,
                            verbose_eval=100)
        model_path = WORK / "models" / f"{source}.json"
        booster.save_model(str(model_path))
        score = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        valid_path = WORK / f"validation_{source}.parquet"
        pq.write_table(pa.table({
            "qid": val.qid.to_numpy(), "tid": val.tid.to_numpy(),
            "y": val.y.to_numpy(dtype=np.uint8), "score": score.astype(np.float32),
        }), valid_path, compression="zstd")
        log(f"{source}: best_iteration={booster.best_iteration}, validation scores saved")
        return str(model_path), str(valid_path)
    finally:
        db.close()
        gc.collect()


# Two independent, source-specific models fit concurrently on cuda:0 and cuda:1.
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(train_one, "source2", 0), pool.submit(train_one, "source3", 1)]
    trained = [f.result() for f in futures]
model_paths = {"source2": Path(trained[0][0]), "source3": Path(trained[1][0])}
validation_paths = [Path(trained[0][1]), Path(trained[1][1])]


validation_db = con()
gt = csv_source(DATA_ROOT / "train" / "train_ground_truth.tsv")
validation_db.execute(f"""
  CREATE TEMP TABLE truth AS
  SELECT source1_entity_id AS qid,
    CASE WHEN matched_entity_ids IS NULL OR matched_entity_ids = ''
      THEN 0 ELSE length(str_split(matched_entity_ids, ',')) END AS true_n
  FROM {gt} WHERE hash(source1_entity_id) % 100 < {VALID_MOD}
""")
validation_db.execute(f"CREATE TEMP VIEW scores AS SELECT * FROM read_parquet({parquet_files(validation_paths)})")
total_true = validation_db.execute("SELECT sum(true_n) FROM truth").fetchone()[0]
recalled = validation_db.execute("SELECT sum(y) FROM scores").fetchone()[0] or 0
candidate_recall = float(recalled / total_true)


def validation_metric(threshold):
    out = validation_db.execute(f"""
      WITH predicted AS (
        SELECT qid, count(*) AS pred_n, sum(y) AS hit_n
        FROM scores WHERE score >= {float(threshold)} GROUP BY qid
      )
      SELECT avg(CASE WHEN true_n = 0 AND coalesce(pred_n,0) = 0 THEN 1.0
        ELSE 1.25 * coalesce(hit_n,0) / (0.25 * true_n + coalesce(pred_n,0)) END),
        sum(CASE WHEN true_n=0 AND coalesce(pred_n,0)=0 THEN 1 ELSE 0 END),
        sum(CASE WHEN true_n=0 THEN 1 ELSE 0 END)
      FROM truth LEFT JOIN predicted USING(qid)
    """).fetchone()
    return {"threshold": float(threshold), "macro_f05": float(out[0]),
            "correct_singletons": int(out[1]), "singletons": int(out[2]),
            "candidate_link_recall": candidate_recall}


grid = np.r_[np.arange(0.05, 0.951, 0.05), [0.975, 0.99, 0.995]]
first = [validation_metric(t) for t in grid]
initial_best = max(first, key=lambda r: (r["macro_f05"], r["threshold"]))
fine = np.arange(max(0.005, initial_best["threshold"] - 0.045),
                 min(0.999, initial_best["threshold"] + 0.046), 0.005)
metrics = first + [validation_metric(t) for t in fine]
best = max(metrics, key=lambda r: (r["macro_f05"], r["threshold"]))
threshold = best["threshold"]
validation_db.close()
log(f"Validation: {best}")
if best["candidate_link_recall"] < 0.98:
    log("WARNING: blocking recall below 98%; inspect missed links before a portal upload.")


def predict_one(source, gpu):
    booster = xgb.Booster()
    booster.load_model(str(model_paths[source]))
    booster.set_param({"device": f"cuda:{gpu}", "nthread": max(2, THREADS // 2)})
    best_iteration = booster.attr("best_iteration")
    prediction_rounds = int(best_iteration) + 1 if best_iteration is not None else booster.num_boosted_rounds()
    source_parts = [p for p in test_parts if p.name.startswith(source + "_")]
    accepted_files = []
    for part in source_parts:
        out = WORK / "accepted" / part.name
        writer = pq.ParquetWriter(out, pa.schema([pa.field("qid", pa.string()), pa.field("tid", pa.string())]), compression="zstd")
        accepted_count = 0
        try:
            reader = pq.ParquetFile(part)
            for batch in reader.iter_batches(batch_size=200_000, columns=["qid", "tid"] + FEATURES):
                frame = batch.to_pandas()
                matrix = xgb.DMatrix(np.ascontiguousarray(frame[FEATURES].to_numpy(dtype=np.float32)), feature_names=FEATURES)
                score = booster.predict(matrix, iteration_range=(0, prediction_rounds))
                keep = score >= threshold
                if keep.any():
                    selected = frame.loc[keep, ["qid", "tid"]]
                    writer.write_table(pa.Table.from_pandas(selected, preserve_index=False))
                    accepted_count += len(selected)
                del frame, matrix, score
            log(f"Inference {source} {part.name}: accepted {accepted_count:,}")
        finally:
            writer.close()
        accepted_files.append(out)
    return accepted_files


with ThreadPoolExecutor(max_workers=2) as pool:
    f2 = pool.submit(predict_one, "source2", 0)
    f3 = pool.submit(predict_one, "source3", 1)
    accepted_parts = f2.result() + f3.result()


def write_wide(parts, out, id_column):
    db = con()
    try:
        test_s1 = csv_source(DATA_ROOT / "test" / "test_source1.tsv")
        files = parquet_files(parts)
        db.execute(f"""
          COPY (
            WITH lists AS (
              SELECT qid, string_agg(tid, ',' ORDER BY tid) AS ids
              FROM read_parquet({files}) GROUP BY qid
            )
            SELECT s.entity_id AS source1_entity_id,
              coalesce(l.ids, '') AS {id_column}
            FROM {test_s1} s LEFT JOIN lists l ON s.entity_id = l.qid
            ORDER BY s.entity_id
          ) TO {sq(out)} (HEADER true, DELIMITER '\\t')
        """)
    finally:
        db.close()


write_wide(test_parts, OUTPUT / "candidate_pairs.tsv", "candidate_entity_ids")
write_wide(accepted_parts, OUTPUT / "matching_results.tsv", "matched_entity_ids")


def validate_wide_streaming():
    candidate = OUTPUT / "candidate_pairs.tsv"
    matching = OUTPUT / "matching_results.tsv"
    expected = 0
    with (DATA_ROOT / "test" / "test_source1.tsv").open(encoding="utf-8", newline="") as f:
        expected = sum(1 for _ in f) - 1
    rows = 0
    with candidate.open(encoding="utf-8", newline="") as cf, matching.open(encoding="utf-8", newline="") as mf:
        creader, mreader = csv.reader(cf, delimiter="\t"), csv.reader(mf, delimiter="\t")
        assert next(creader) == ["source1_entity_id", "candidate_entity_ids"]
        assert next(mreader) == ["source1_entity_id", "matched_entity_ids"]
        previous = ""
        for cr, mr in zip(creader, mreader):
            assert len(cr) == len(mr) == 2
            assert cr[0] == mr[0] and cr[0] > previous and cr[0].startswith("S1-")
            cids = cr[1].split(",") if cr[1] else []
            mids = mr[1].split(",") if mr[1] else []
            assert len(cids) == len(set(cids)) and len(mids) == len(set(mids))
            assert all(x.startswith(("S2-", "S3-")) for x in cids)
            assert set(mids).issubset(cids)
            previous = cr[0]
            rows += 1
        assert next(creader, None) is None and next(mreader, None) is None
    assert rows == expected, (rows, expected)
    log(f"Streaming output validation passed: {rows:,} unique Source 1 rows; matches subset of candidates")


validate_wide_streaming()
official_validator = DATA_ROOT.parent / "utils" / "validate_submission.py"
if official_validator.exists():
    # The supplied validator allows matching-only mode. Its candidate mode holds
    # tens of millions of Python strings in RAM; streaming check above covers both.
    subprocess.check_call([sys.executable, str(official_validator),
                           "--matching", str(OUTPUT / "matching_results.tsv"),
                           "--test-dir", str(DATA_ROOT / "test")])

manifest = {
    "data_root": str(DATA_ROOT), "country_labels_train": train_countries,
    "country_labels_test": test_countries, "validation": best,
    "model_license": "Apache-2.0 (XGBoost)", "xgboost": xgb.__version__,
    "duckdb": duckdb.__version__, "pandas": pd.__version__,
    "pyarrow": pa.__version__, "torch": torch.__version__,
    "gpu_names": [torch.cuda.get_device_name(i) for i in (0, 1)],
    "query_sample_mod": QUERY_SAMPLE_MOD, "valid_mod": VALID_MOD,
    "max_token_df": MAX_TOKEN_DF, "max_exact_df": MAX_EXACT_DF,
    "max_token_keys_per_channel": MAX_TOKEN_KEYS_PER_CHANNEL,
    "max_candidates_per_source": MAX_CANDIDATES_PER_SOURCE,
    "output": {p.name: {"bytes": p.stat().st_size,
                          "sha256": sha256_stream(p)}
               for p in (OUTPUT / "matching_results.tsv", OUTPUT / "candidate_pairs.tsv")},
}
(WORK / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
log(f"Finished. Upload only {OUTPUT / 'matching_results.tsv'} to the portal after review.")
log(f"Final package also needs {OUTPUT / 'candidate_pairs.tsv'} and runnable code.")
