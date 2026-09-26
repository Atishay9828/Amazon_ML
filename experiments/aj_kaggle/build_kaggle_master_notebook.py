"""Build the standalone Kaggle-style full-scale entity-resolution notebook.

The notebook embeds reviewed source modules from origin/feat/data-candidates
so it remains self-contained after export to Kaggle.
"""

from __future__ import annotations

import subprocess
import textwrap
import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("kaggle_master_style_entity_resolution.ipynb")
SOURCE_REF = "09bb350a7dd52393cf8d27e002c8ffdf67840685"


def git_text(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{SOURCE_REF}:{path}"], cwd=ROOT,
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip("\n"))


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip("\n"))


data_source = git_text("code/business_entity_resolution/src/data.py")
old_reader = 'with path.open("r", encoding="utf-8-sig", newline="") as handle:'
new_reader = '''if path.suffix == ".gz":
        import gzip
        handle_context = gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    else:
        handle_context = path.open("r", encoding="utf-8-sig", newline="")
    with handle_context as handle:'''
if data_source.count(old_reader) != 2:
    raise RuntimeError("The upstream TSV reader changed; review gzip patch before rebuilding.")
data_source = data_source.replace(old_reader, new_reader)

normalization_source = git_text("code/business_entity_resolution/src/normalization.py")
candidate_source = git_text("code/business_entity_resolution/src/candidates.py")
assembly_source = (Path(__file__).with_name("output_assembly.py")).read_text(encoding="utf-8")
license_text = git_text("LICENSE")
BUILD_CODE_SHA256 = hashlib.sha256(b"\0".join([
    Path(__file__).read_bytes(), data_source.encode("utf-8"),
    normalization_source.encode("utf-8"), candidate_source.encode("utf-8"),
    assembly_source.encode("utf-8"),
])).hexdigest()


cells = [
    markdown(r"""
    # Full-scale Kaggle-style entity resolution (dual T4)

    **Before running:** attach the official `student_resource` dataset, select **GPU T4 x2**, restart the session, then run the notebook from top to bottom. Internet is used only if the small `anyascii` text-normalization dependency is missing from Kaggle's image. Keep `/kaggle/working` intact to resume in the same session. After a session reset, save a notebook version with output files and attach that version as an input; setup looks for a matching checkpoint and restores it when space permits.

    This notebook combines the strongest pieces we already have with the broader multi-route candidate search from the team branch. It searches the complete target databases, generates candidates for every test Source 1 row, trains fresh pair matchers, selects a threshold on a clean validation group, checks that choice once on a separate holdout, and writes both required TSV files.

    **Model plan:** broad token + character retrieval; pairwise name/address, numeric, cosine, and route-channel features; separate XGBoost GPU matchers for Source 2 and Source 3; LightGBM as a challenger when installed; threshold selection on the challenge's macro F0.5 score. The best validation model or simple XGBoost/LightGBM blend is selected, then scored against all test candidates. There is no global one-to-one assignment. The published first-place Foursquare system also used CatBoost and multilingual transformers; those are follow-up options here because this dataset has only name/address text and those models would add a separate encoder/data pipeline before we know they improve the clean holdout.

    **Data scope:** every Source 2 and Source 3 record is searchable for each query. To keep training and storage bounded, this run fits on a deterministic 40% Source 1 query cohort, tunes on 5%, and reports one untouched 5% holdout. That is four times the earlier 10% fit split; all test Source 1 rows are still processed. For fitting only, it keeps every positive pair plus the 16 most text-similar negatives per query. Validation and holdout retain every retrieved pair, so model/threshold selection and reporting use the real candidate distribution. Queries used to choose retrieval settings are excluded from fit, validation, and holdout.

    **Expected run time:** highly uncertain. The completed 10k-query broad-retrieval run took about 32.5 minutes at cap 64. A straight query-count extrapolation to roughly 2.8m train-plus-test queries is around **150 hours**, but that is not a reliable prediction because country indexing, cache reuse, concurrency, and feature/output work do not scale linearly. The 128-candidate budget adds downstream work. Use country progress logs to replace this estimate with observed throughput. Kaggle documents a 12-hour CPU/GPU session, so continuing may require saved output checkpoints across sessions; a matching checkpoint restores only when Kaggle exposes it as an attached input and working disk has room.

    | Evidence | Reported time | What it covers |
    |---|---:|---|
    | Measured broad-token retrieval (K32/A16, cap 64) on 10k development queries | 32.5 min; 93.74% candidate-link recall; 97.70% oracle macro F0.5; about 20 GiB parent RSS | Full target search, sampled queries only; not matcher score and not the wider K64/A32, cap-128 configuration below |
    | Older retrieval setup on 551,705 training queries | about 68–69 min per shard | Different, less broad configuration; not a safe full-run extrapolation |
    | Foursquare 7th-place team | not reported in its writeup | Multiple retrieval routes and a LightGBM pair classifier |
    | Foursquare 11th-place team | 24–48 h embedding training; under 1 h for XGBoost | BERT metric-learning system with location features on external compute; not comparable to this text-only run |
    | Foursquare 1st-place team | total time not reported | Reported model families/epochs and hardware, but no end-to-end wall time |
    | This notebook | ~150 h naive query-count extrapolation; very low confidence | 40% train-query fit cohort plus all test queries, wider retrieval, features, bakeoff, inference, and output assembly |

    **Kaggle winner evidence:** the 7th-place Foursquare solution used multilingual text embeddings, multiple candidate routes, and a LightGBM pair classifier; its writeup gives no total elapsed time. The 11th-place team reported 24–48 hours for 40-epoch BERT embedding training on external compute and under one hour for its XGBoost models on Kaggle. That is a different multimodal task with latitude/longitude, and the writeup discusses leakage, so it is only a rough runtime reference. The 1st-place team stacked LightGBM/CatBoost pair models with multilingual transformers and graph post-processing; it also gives no total runtime and explicitly discusses train/test leakage. This notebook transfers only reusable, label-safe retrieval and pairwise-modeling ideas; it does not copy leakage or location features unavailable here.

    The official Foursquare writeups are historical examples, not a guarantee that the same models will win this task: [7th-place solution](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/in-tokyo-7th-place-solution-with-inference-code) · [11th-place solution](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/mc-digital-11th-place-code-and-solution) · [1st-place solution](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution). Kaggle's current notebook documentation lists a 12-hour CPU/GPU session and 20 GB of auto-saved working disk: [Kaggle notebook limits](https://www.kaggle.com/docs/notebooks).

    **Evidence status:** not run here. This workspace has no attached official challenge dataset or Kaggle GPU. The cap-64 development run measured a 97.70% retrieval-oracle macro F0.5 on 10k queries; that candidate set cannot support the 98.9% leaderboard target. The wider cap-128 setup is a measured-quality hypothesis, not a score guarantee. Its clean validation oracle and fitted macro F0.5 are the first decision gates.
    """),
    code(r"""
    # Kaggle setup: do not replace Kaggle's CUDA-enabled XGBoost build.
    import importlib.util
    import os
    import subprocess
    import sys
    from pathlib import Path

    ENGINE_DIR = Path("/kaggle/working/amazon_ml_master_engine")
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)

    retrieval_packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "anyascii": "anyascii==0.3.3",
    }
    missing = [package for module, package in retrieval_packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print("Installing only missing CPU retrieval dependencies:", missing, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import xgboost as xgb

    try:
        import lightgbm as lgb
        HAS_LIGHTGBM = True
    except ImportError:
        lgb = None
        HAS_LIGHTGBM = False

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Select Kaggle GPU T4 x2, then restart and run all cells.")
    print({
        "python": sys.version.split()[0], "duckdb": duckdb.__version__,
        "xgboost": xgb.__version__, "torch": torch.__version__,
        "lightgbm_available": HAS_LIGHTGBM,
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
    }, flush=True)
    """),
    code("%%writefile /kaggle/working/amazon_ml_master_engine/data.py\n" + data_source),
    code("%%writefile /kaggle/working/amazon_ml_master_engine/normalization.py\n" + normalization_source),
    code("%%writefile /kaggle/working/amazon_ml_master_engine/candidates.py\n" + candidate_source),
    code("%%writefile /kaggle/working/amazon_ml_master_engine/output_assembly.py\n" + assembly_source),
    markdown(r"""
    ## 1. Locate the official files and freeze this run

    The candidate engine is copied from team branch `feat/data-candidates` at commit `09bb350a7dd52393cf8d27e002c8ffdf67840685` (Apache-2.0). This notebook uses K64/A32 retrieval, a 256-row per-source staged shortlist, and a 128-row combined final set; its TSV reader accepts the gzip country shards written below. The complete upstream Apache-2.0 license is included in the final notebook cell.
    """),
    code(r"""
    import csv
    import gc
    import gzip
    import hashlib
    import json
    import os
    import random
    import shutil
    import sys
    import time
    from collections import defaultdict
    from pathlib import Path

    sys.path.insert(0, str(ENGINE_DIR))
    import candidates as retrieval
    import data as source_data
    from output_assembly import (
        ao_atomic_json, ao_connection, ao_file_list, ao_fingerprint, ao_log,
        ao_parquet_info, ao_resources, ao_sha256, assemble_outputs,
    )

    input_roots = sorted({
        path.parent.parent.resolve()
        for path in Path("/kaggle/input").glob("**/dataset/train/train_source1.tsv")
    })
    if len(input_roots) != 1:
        raise RuntimeError(f"Attach exactly one official dataset; found {input_roots}")
    DATA_ROOT = input_roots[0]
    REQUIRED = [
        DATA_ROOT / split / f"{split}_source{source}.tsv"
        for split in ("train", "test") for source in (1, 2, 3)
    ] + [DATA_ROOT / "train" / "train_ground_truth.tsv"]
    missing = [str(path) for path in REQUIRED if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Official challenge files are missing: {missing}")

    def file_id(path):
        stat = Path(path).stat()
        return {"name": Path(path).name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    # Content hashes make a saved checkpoint specific to the exact official files,
    # and relative keys keep its identity stable if Kaggle changes the mount name.
    INPUT_IDS = {
        path.relative_to(DATA_ROOT).as_posix(): {
            "bytes": path.stat().st_size, "sha256": ao_sha256(path),
        }
        for path in REQUIRED
    }

    # Broad-token routes from the completed 10k full-target experiment, with
    # a wider staged shortlist and final cap to raise the retrieval ceiling.
    SETTINGS = retrieval.Settings(
        name_k=64, address_k=32, cap=128, stage_cap=256,
        indexed_grams=16, query_grams=32, gram_selection="rarest",
        pair_max_df=5_000, pair_max_hits=2_000,
        single_max_df=512, single_tokens=2, workers=2,
    )
    TARGET_MACRO_F05 = 0.989
    FORCE_TEST_RUN_BELOW_TARGET = False
    QUERY_POLICY = {
        "stable_hash": "sha256(qid) first 8 hex digits modulo 100",
        "valid_buckets": "0-4", "holdout_buckets": "5-9",
        "fit_buckets": "10-49", "previously_inspected_sample_excluded": True,
    }
    FIT_NEGATIVES_PER_QUERY = 16
    TRAINING_POLICY = {
        "fit_negatives_per_query": FIT_NEGATIVES_PER_QUERY,
        "fit_negative_selection": "highest max(name/address Jaro-Winkler and cosine), tid tiebreak",
        "fit_query_weight": "each query's retained pair weights sum to one",
        "validation_and_holdout": "all retrieved pairs, no negative sampling",
    }
    FEATURE_PIPELINE_VERSION = "duckdb-pair-features-v1"
    XGBOOST_PARAMS = {
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "tree_method": "hist", "max_depth": 8, "eta": 0.07,
        "max_bin": 256, "subsample": 0.85, "colsample_bytree": 0.9,
        "min_child_weight": 8, "lambda": 4.0,
    }
    XGBOOST_TRAINING = {"num_boost_round": 1200, "early_stopping_rounds": 75}
    LIGHTGBM_PARAMS = {
        "objective": "binary", "n_estimators": 1500, "learning_rate": 0.05,
        "num_leaves": 63, "subsample": 0.85, "subsample_freq": 1,
        "colsample_bytree": 0.9, "reg_lambda": 4.0,
        "min_child_samples": 50, "verbosity": -1,
    }
    LIGHTGBM_TRAINING = {"early_stopping_rounds": 75, "eval_metric": "binary_logloss"}
    MODEL_CONFIGURATION = {
        "xgboost": XGBOOST_PARAMS, "xgboost_training": XGBOOST_TRAINING,
        "lightgbm": LIGHTGBM_PARAMS, "lightgbm_training": LIGHTGBM_TRAINING,
        "source_seed_rule": "2026 + source number",
    }
    RUNTIME_VERSIONS = {
        "duckdb": duckdb.__version__, "xgboost": xgb.__version__,
        "lightgbm": getattr(lgb, "__version__", None),
        "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__,
        "torch": torch.__version__,
    }
    BUILD_CODE_SHA256 = "BUILD_CODE_SHA256_PLACEHOLDER"
    PIPELINE_VERSION = "kaggle-master-style-v3"
    run_identity = {
        "version": PIPELINE_VERSION, "build_code_sha256": BUILD_CODE_SHA256,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "model_configuration": MODEL_CONFIGURATION, "runtime_versions": RUNTIME_VERSIONS,
        "source_commit": "09bb350a7dd52393cf8d27e002c8ffdf67840685",
        "settings": vars(SETTINGS), "query_policy": QUERY_POLICY,
        "training_policy": TRAINING_POLICY, "inputs": INPUT_IDS,
    }
    RUN_ID = hashlib.sha256(json.dumps(run_identity, sort_keys=True).encode()).hexdigest()[:12]
    WORK = Path("/kaggle/working") / f"amazon_ml_master_{RUN_ID}"
    checkpoint_inputs = [
        path for path in Path("/kaggle/input").glob(f"**/amazon_ml_master_{RUN_ID}")
        if (path / "run_identity.json").is_file()
    ]
    saved_identity = WORK / "run_identity.json"
    if saved_identity.is_file():
        if json.loads(saved_identity.read_text(encoding="utf-8")) != run_identity:
            raise RuntimeError(f"Existing work directory has a different run identity: {WORK}")
    elif checkpoint_inputs:
        if len(checkpoint_inputs) != 1:
            raise RuntimeError(f"Found {len(checkpoint_inputs)} matching attached checkpoints; attach only one")
        checkpoint = checkpoint_inputs[0]
        checkpoint_identity = json.loads((checkpoint / "run_identity.json").read_text(encoding="utf-8"))
        if checkpoint_identity != run_identity:
            raise RuntimeError("Attached checkpoint content identity does not match the official input files")
        def ignore_checkpoint_cache(directory, names):
            return {name for name in names
                    if name == "tmp" or ".partial" in name.lower() or name.lower().endswith(".tmp")}
        checkpoint_bytes = sum(
            item.stat().st_size for item in checkpoint.rglob("*")
            if item.is_file() and ".partial" not in item.name.lower()
            and not item.name.lower().endswith(".tmp")
            and "tmp" not in item.relative_to(checkpoint).parts
        )
        free_bytes = shutil.disk_usage(Path("/kaggle/working")).free
        if checkpoint_bytes > free_bytes * 0.85:
            raise RuntimeError(
                f"Matching checkpoint needs about {checkpoint_bytes / 1024**3:.1f} GiB, "
                f"but only {free_bytes / 1024**3:.1f} GiB is free under /kaggle/working. "
                "Free disk or attach a smaller completed checkpoint before resuming."
            )
        shutil.copytree(checkpoint, WORK, dirs_exist_ok=True, ignore=ignore_checkpoint_cache)
        print(f"Restored matching checkpoint from {checkpoint}", flush=True)
    for folder in ("target_shards", "candidate_files", "retrieval", "train_features",
                   "test_features", "models", "accepted", "tmp", "output"):
        (WORK / folder).mkdir(parents=True, exist_ok=True)
    (WORK / "run_identity.json").write_text(json.dumps(run_identity, indent=2), encoding="utf-8")
    OUTPUT = WORK / "output"
    THREADS = max(2, min(8, os.cpu_count() or 4))

    print(f"Official files found. Run ID: {RUN_ID}; work directory: {WORK}")
    print(f"Free disk now: {shutil.disk_usage(WORK).free / 1024**3:.1f} GiB")
    """),
    markdown(r"""
    ## 2. Make compact country target shards

    Country equality is supported by the observed training labels, which showed no cross-country true links. We split each target source into gzip shards in one streaming pass, so each query is ranked only against same-country records and the candidate cap is not consumed by unrelated countries. France has no labeled training examples, so cross-country blocking remains an unverified assumption for France. The complete source files remain the authoritative inputs for feature joins and final validation.
    """),
    code(r"""
    SOURCE_COLUMNS = list(source_data.SOURCE_COLUMNS)

    def country_key(country):
        return hashlib.sha1(country.encode("utf-8")).hexdigest()[:12]

    def target_shard_path(split, source, country):
        return WORK / "target_shards" / split / f"source{source}_{country_key(country)}.tsv.gz"

    def feature_files_ready(split):
        suffix = "fit" if split == "train" else "all"
        for source in (2, 3):
            path = WORK / f"{split}_features" / f"source{source}_{suffix}.parquet"
            marker = path.with_suffix(".complete.json")
            if not path.is_file() or not marker.is_file():
                return False
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if (saved.get("identity", {}).get("run_id") != RUN_ID
                    or saved.get("sha256") != ao_sha256(path)):
                return False
        return True

    def partition_target_file(split, source):
        raw_path = DATA_ROOT / split / f"{split}_source{source}.tsv"
        out_dir = WORK / "target_shards" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / f"source{source}.complete.json"
        identity = {"input": file_id(raw_path), "source": source, "split": split,
                    "columns": SOURCE_COLUMNS, "gzip": True}
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("identity") == identity:
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if all(Path(item).is_file() for item in saved.get("shards", [])):
                return saved["countries"]

        handles, writers, counts = {}, {}, defaultdict(int)
        temp_paths = {}
        with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t", strict=True)
            header = next(reader, None)
            if header != SOURCE_COLUMNS:
                raise ValueError(f"Unexpected columns in {raw_path}: {header}")
            for line_number, row in enumerate(reader, start=2):
                if len(row) != 4 or not row[3].strip():
                    raise ValueError(f"Bad row or empty country at {raw_path}:{line_number}")
                if not row[0].startswith(f"S{source}-"):
                    raise ValueError(f"Wrong source ID at {raw_path}:{line_number}: {row[0]}")
                country = row[3]
                key = country_key(country)
                if country not in writers:
                    final_path = out_dir / f"source{source}_{key}.tsv.gz"
                    temp_path = final_path.with_name(final_path.name + ".partial")
                    handle = gzip.open(temp_path, "wt", encoding="utf-8", newline="", compresslevel=3)
                    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                    writer.writerow(SOURCE_COLUMNS)
                    handles[country] = handle
                    writers[country] = writer
                    temp_paths[country] = (temp_path, final_path)
                writers[country].writerow(row)
                counts[country] += 1
        for handle in handles.values():
            handle.close()
        for temp_path, final_path in temp_paths.values():
            os.replace(temp_path, final_path)
        saved = {
            "identity": identity, "countries": sorted(counts),
            "row_counts": dict(counts),
            "shards": [str(target_shard_path(split, source, country)) for country in sorted(counts)],
        }
        ao_atomic_json(marker, saved)
        print(f"Target shards {split} S{source}: {sum(counts.values()):,} rows across {len(counts)} countries")
        return saved["countries"]

    target_countries = {}
    if feature_files_ready("train"):
        print("Reuse verified train feature files; skip train target-shard rebuild", flush=True)
    else:
        for source in (2, 3):
            target_countries[("train", source)] = set(partition_target_file("train", source))
    """),
    markdown(r"""
    ## 3. Generate checkpointed candidates

    The training queries are selected deterministically. The 10k retrieval-development sample and the next 50k inspected queries from the earlier experiment are excluded entirely. Each country/source writes completed compressed candidates only after its retrieval chunks finish; rerunning this cell reuses completed work and resumes missing chunks. The final cap is 128 combined across Sources 2 and 3, with up to 256 ranked proposals staged per source first.
    """),
    code(r"""
    def sha_bucket(entity_id):
        return int(hashlib.sha256(entity_id.encode("utf-8")).hexdigest()[:8], 16) % 100

    def load_train_query_cohorts():
        records = retrieval._load_sorted_source1(source_data.source_path(DATA_ROOT, "train", 1), None)
        sampled = random.Random(20260926).sample(range(len(records)), min(60_000, len(records)))
        inspected_ids = {records[index].entity_id for index in sampled}
        grouped, scope_rows = defaultdict(list), []
        for record in records:
            if record.entity_id in inspected_ids:
                continue
            bucket = sha_bucket(record.entity_id)
            cohort = "valid" if bucket < 5 else "holdout" if bucket < 10 else "fit" if bucket < 50 else None
            if cohort:
                grouped[record.country].append(record)
                scope_rows.append((record.entity_id, cohort))
        scope_rows.sort()
        scope_path = WORK / "train_query_scope.tsv"
        with scope_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["qid", "cohort"])
            writer.writerows(scope_rows)
        del records
        gc.collect()
        print({"selected_train_queries": len(scope_rows),
               "fit": sum(cohort == "fit" for _, cohort in scope_rows),
               "valid": sum(cohort == "valid" for _, cohort in scope_rows),
               "holdout": sum(cohort == "holdout" for _, cohort in scope_rows),
               "previously_inspected_excluded": len(inspected_ids)})
        return dict(grouped), scope_path, len(scope_rows)

    def load_test_queries():
        records = retrieval._load_sorted_source1(source_data.source_path(DATA_ROOT, "test", 1), None)
        grouped = defaultdict(list)
        for record in records:
            grouped[record.country].append(record)
        print(f"Test Source 1 queries: {len(records):,} across {len(grouped)} countries")
        del records
        gc.collect()
        return dict(grouped)

    TRAIN_QUERIES, SCOPE_PATH, TRAIN_QUERY_COUNT = load_train_query_cohorts()

    def query_digest(records):
        digest = hashlib.sha256()
        for record in records:
            digest.update(record.entity_id.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def write_country_candidates(split, country, records):
        key = country_key(country)
        output_dir = WORK / "candidate_files" / split
        output_dir.mkdir(parents=True, exist_ok=True)
        out2 = output_dir / f"source2_{key}.tsv.gz"
        out3 = output_dir / f"source3_{key}.tsv.gz"
        marker = output_dir / f"country_{key}.complete.json"
        country_work = WORK / "retrieval" / split / key
        country_work.mkdir(parents=True, exist_ok=True)
        target_paths = {source: target_shard_path(split, source, country) for source in (2, 3)}
        identity = {
            "run_id": RUN_ID, "split": split, "country": country,
            "query_count": len(records), "query_sha256": query_digest(records),
            "settings": vars(SETTINGS),
            "targets": {str(source): file_id(path) if path.is_file() else None
                        for source, path in target_paths.items()},
        }
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("identity") == identity:
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if out2.is_file() and out3.is_file() and ao_sha256(out2) == saved.get("source2_sha256") and ao_sha256(out3) == saved.get("source3_sha256"):
                return saved

        for source in (2, 3):
            path = target_paths[source]
            if path.is_file() and path.stat().st_size > 40:
                retrieval._retrieve_source(records, path, source, SETTINGS, country_work, None)

        temp2 = out2.with_name(out2.name + ".partial")
        temp3 = out3.with_name(out3.name + ".partial")
        counts = {2: 0, 3: 0}
        with gzip.open(temp2, "wt", encoding="utf-8", newline="", compresslevel=3) as h2, \
             gzip.open(temp3, "wt", encoding="utf-8", newline="", compresslevel=3) as h3:
            writer2, writer3 = csv.writer(h2, delimiter="\t", lineterminator="\n"), csv.writer(h3, delimiter="\t", lineterminator="\n")
            writer2.writerow(retrieval.HEADER)
            writer3.writerow(retrieval.HEADER)
            chunks = (len(records) + SETTINGS.query_chunk - 1) // SETTINGS.query_chunk
            for chunk in range(chunks):
                p2 = retrieval._part_path(country_work, 2, chunk)
                p3 = retrieval._part_path(country_work, 3, chunk)
                groups2 = retrieval._read_part(p2) if p2.is_file() else {}
                groups3 = retrieval._read_part(p3) if p3.is_file() else {}
                batch = records[chunk * SETTINGS.query_chunk:(chunk + 1) * SETTINGS.query_chunk]
                for record in batch:
                    selected = retrieval._choose_rows(
                        groups2.get(record.entity_id, []), groups3.get(record.entity_id, []), SETTINGS.cap
                    )
                    for row in selected:
                        if row[1].startswith("S2-"):
                            writer2.writerow(row)
                            counts[2] += 1
                        elif row[1].startswith("S3-"):
                            writer3.writerow(row)
                            counts[3] += 1
                        else:
                            raise RuntimeError(f"Unexpected target ID: {row[1]}")
                del groups2, groups3
        os.replace(temp2, out2)
        os.replace(temp3, out3)
        saved = {
            "identity": identity, "source1_queries": len(records),
            "source2_pairs": counts[2], "source3_pairs": counts[3],
            "source2_sha256": ao_sha256(out2), "source3_sha256": ao_sha256(out3),
        }
        ao_atomic_json(marker, saved)
        print(f"Candidates {split} {country}: S2={counts[2]:,}; S3={counts[3]:,}; cap={SETTINGS.cap}", flush=True)
        return saved

    def run_candidate_split(split, grouped):
        feature_dir = WORK / f"{split}_features"
        feature_suffix = "fit" if split == "train" else "all"
        feature_ready = True
        for source in (2, 3):
            feature_path = feature_dir / f"source{source}_{feature_suffix}.parquet"
            feature_marker = feature_path.with_suffix(".complete.json")
            if not feature_path.is_file() or not feature_marker.is_file():
                feature_ready = False
                break
            saved = json.loads(feature_marker.read_text(encoding="utf-8"))
            if (saved.get("identity", {}).get("run_id") != RUN_ID
                    or saved.get("sha256") != ao_sha256(feature_path)):
                feature_ready = False
                break
        if feature_ready:
            print(f"Reuse completed {split} feature files; raw country candidate files are no longer needed.")
            return
        started = time.monotonic()
        countries = sorted(grouped)
        total_queries = sum(len(grouped[country]) for country in countries)
        processed_queries = 0
        for i, country in enumerate(countries, start=1):
            write_country_candidates(split, country, grouped[country])
            processed_queries += len(grouped[country])
            if i == 1 or i % 5 == 0 or i == len(countries):
                elapsed = max(1.0, time.monotonic() - started)
                rate = processed_queries / elapsed
                eta_hours = (total_queries - processed_queries) / max(rate, 1e-9) / 3600
                print(f"{split}: countries {i}/{len(countries)}; queries {processed_queries:,}/{total_queries:,}; "
                      f"elapsed {elapsed/3600:.2f}h; rough remaining {eta_hours:.2f}h", flush=True)
        candidate_files = sorted((WORK / "candidate_files" / split).glob("source[23]_*.tsv.gz"))
        print(f"{split}: {len(candidate_files)} compressed candidate files; free disk {shutil.disk_usage(WORK).free / 1024**3:.1f} GiB")

    """),
    markdown(r"""
    ## 4. Recompute pair features on the exact candidate set

    Candidate generation and ranking are separated. The features below are calculated from the original official source files after joining only the selected candidate IDs. This avoids training on one candidate distribution and submitting another. The XGBoost/LightGBM models are trained from scratch on this broader candidate set.

    Only train retrieval and features are built in this section. Test retrieval and features are deferred until validation and the untouched holdout establish that the measured retrieval upper bound can plausibly reach the 98.9% target.
    """),
    code(r'''
    import duckdb

    CHANNELS = [
        "exact_name", "exact_address", "name_signature", "compact_name",
        "rare_name", "rare_address", "name_pair", "address_pair", "cross_token",
        "single_name", "single_address", "name_char", "address_char",
    ]
    FEATURES = [
        "name_jw", "address_jw", "name_jaccard", "address_jaccard",
        "name_length_ratio", "address_length_ratio", "number_equal",
        "number_both_present", "exact_name", "exact_address",
        "name_cosine", "address_cosine", "channel_count",
    ] + [f"channel_{channel}" for channel in CHANNELS]

    def sql_quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    def csv_relation(path):
        return f"read_csv({sql_quote(path)}, delim='\\t', header=true, all_varchar=true)"

    def candidate_relation(paths):
        if not paths:
            raise RuntimeError("No candidate files found for a source")
        file_list = "[" + ",".join(sql_quote(path) for path in paths) + "]"
        return f"read_csv({file_list}, delim='\\t', header=true, all_varchar=true, union_by_name=true)"

    def feature_connection():
        db = duckdb.connect()
        db.execute("SET memory_limit='8GB'")
        db.execute(f"SET threads={min(8, THREADS)}")
        db.execute(f"SET temp_directory={sql_quote(WORK / 'tmp')}")
        db.execute("SET preserve_insertion_order=false")
        return db

    def feature_query(split, source, candidate_files, output_path):
        query_path = DATA_ROOT / split / f"{split}_source1.tsv"
        target_path = DATA_ROOT / split / f"{split}_source{source}.tsv"
        cohort_join = ""
        cohort_source_select = ""
        cohort_select = ""
        cohort_filter = ""
        if split == "train":
            cohort_join = f"LEFT JOIN {csv_relation(SCOPE_PATH)} scope ON scope.qid=c.qid"
            cohort_source_select = ", scope.cohort AS cohort"
            cohort_select = ", cohort"
            cohort_filter = "AND scope.cohort IS NOT NULL"

        channel_select = ",\n".join(
            f"CASE WHEN strpos('|' || c.retrieval_channels || '|', '|{name}|') > 0 THEN 1 ELSE 0 END::TINYINT AS channel_{name}"
            for name in CHANNELS
        )
        query = f"""
        COPY (
          WITH c AS (
            SELECT source1_entity_id AS qid, candidate_entity_id AS tid,
                   CAST(name_cosine AS FLOAT) AS name_cosine,
                   CAST(address_cosine AS FLOAT) AS address_cosine,
                   retrieval_channels
            FROM {candidate_relation(candidate_files)}
            WHERE starts_with(candidate_entity_id, {sql_quote('S' + str(source) + '-')})
          ), q AS (
            SELECT entity_id AS rid, country,
              trim(regexp_replace(regexp_replace(
                ' ' || trim(regexp_replace(lower(strip_accents(coalesce(business_name,''))), '[^a-z0-9]+', ' ', 'g')) || ' ',
                ' (incorporated|inc|corporation|corp|limited|ltd|llc|llp|private|pvt) ', ' ', 'g'),
                ' +', ' ', 'g')) AS nm,
              trim(regexp_replace(lower(strip_accents(coalesce(business_address,''))), '[^a-z0-9]+', ' ', 'g')) AS ad,
              regexp_extract(coalesce(business_address,''), '[0-9]+') AS num
            FROM {csv_relation(query_path)}
          ), t AS (
            SELECT entity_id AS rid, country,
              trim(regexp_replace(regexp_replace(
                ' ' || trim(regexp_replace(lower(strip_accents(coalesce(business_name,''))), '[^a-z0-9]+', ' ', 'g')) || ' ',
                ' (incorporated|inc|corporation|corp|limited|ltd|llc|llp|private|pvt) ', ' ', 'g'),
                ' +', ' ', 'g')) AS nm,
              trim(regexp_replace(lower(strip_accents(coalesce(business_address,''))), '[^a-z0-9]+', ' ', 'g')) AS ad,
              regexp_extract(coalesce(business_address,''), '[0-9]+') AS num
            FROM {csv_relation(target_path)}
          ), joined AS (
            SELECT c.qid, c.tid, c.name_cosine, c.address_cosine, c.retrieval_channels,
                   q.nm AS qnm, q.ad AS qad, q.num AS qnum,
                   t.nm AS tnm, t.ad AS tad, t.num AS tnum
                   {cohort_source_select}
            FROM c JOIN q ON q.rid=c.qid JOIN t ON t.rid=c.tid AND t.country=q.country
            {cohort_join}
            WHERE true {cohort_filter}
          )
          SELECT qid, tid,
            jaro_winkler_similarity(qnm, tnm)::FLOAT AS name_jw,
            CASE WHEN qad='' OR tad='' THEN 0 ELSE jaro_winkler_similarity(qad,tad) END::FLOAT AS address_jw,
            CASE WHEN qnm='' OR tnm='' THEN 0 ELSE jaccard(qnm,tnm) END::FLOAT AS name_jaccard,
            CASE WHEN qad='' OR tad='' THEN 0 ELSE jaccard(qad,tad) END::FLOAT AS address_jaccard,
            (least(length(qnm),length(tnm))::FLOAT / greatest(1,length(qnm),length(tnm)))::FLOAT AS name_length_ratio,
            (least(length(qad),length(tad))::FLOAT / greatest(1,length(qad),length(tad)))::FLOAT AS address_length_ratio,
            CASE WHEN qnum<>'' AND qnum=tnum THEN 1 ELSE 0 END::FLOAT AS number_equal,
            CASE WHEN qnum<>'' AND tnum<>'' THEN 1 ELSE 0 END::FLOAT AS number_both_present,
            CASE WHEN qnm<>'' AND qnm=tnm THEN 1 ELSE 0 END::FLOAT AS exact_name,
            CASE WHEN qad<>'' AND qad=tad THEN 1 ELSE 0 END::FLOAT AS exact_address,
            name_cosine, address_cosine,
            (length(retrieval_channels)-length(replace(retrieval_channels,'|',''))
              + CASE WHEN retrieval_channels='' THEN 0 ELSE 1 END)::FLOAT AS channel_count,
            {channel_select}
            {cohort_select}
          FROM joined c
        ) TO {sql_quote(output_path)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)
        """
        return query

    FEATURE_FILES = {"train": {}, "test": {}}
    FEATURE_MARKERS = {"train": {}, "test": {}}

    def build_feature_split(split, query_groups=None):
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if not feature_files_ready(split):
            for source in (2, 3):
                partition_target_file(split, source)
            if query_groups is None:
                query_groups = TRAIN_QUERIES if split == "train" else load_test_queries()
            run_candidate_split(split, query_groups)
            del query_groups
        for source in (2, 3):
            files = sorted((WORK / "candidate_files" / split).glob(f"source{source}_*.tsv.gz"))
            output_dir = WORK / f"{split}_features"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"source{source}_{'fit' if split == 'train' else 'all'}.parquet"
            marker = output_path.with_suffix(".complete.json")
            prior = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
            if (output_path.is_file() and prior.get("identity", {}).get("run_id") == RUN_ID
                    and prior.get("sha256") == ao_sha256(output_path)):
                ao_parquet_info(output_path, ["qid", "tid", *FEATURES])
                FEATURE_FILES[split][source] = output_path
                FEATURE_MARKERS[split][source] = marker
                for candidate_file in files:
                    candidate_file.unlink(missing_ok=True)
                print(f"Reuse features: {output_path.name}; remove regenerated candidate intermediates")
                continue
            candidate_identity = [(str(p), p.stat().st_size, ao_sha256(p)) for p in files]
            identity = {"run_id": RUN_ID, "split": split, "source": source,
                        "feature_names": FEATURES, "candidates": candidate_identity}
            free_gb = shutil.disk_usage(WORK).free / 1024**3
            print(f"Build {split} S{source} features from {len(files)} country files; free disk={free_gb:.1f} GiB", flush=True)
            if output_path.exists():
                output_path.unlink()
            db = feature_connection()
            try:
                db.execute(feature_query(split, source, [str(p) for p in files], output_path))
            finally:
                db.close()
                gc.collect()
            rows = ao_parquet_info(output_path, ["qid", "tid", *FEATURES])
            ao_atomic_json(marker, {"identity": identity, "rows": rows, "sha256": ao_sha256(output_path)})
            FEATURE_FILES[split][source] = output_path
            FEATURE_MARKERS[split][source] = marker
            for candidate_file in files:
                candidate_file.unlink(missing_ok=True)
            print(f"Feature rows {split} S{source}: {rows:,}; file={output_path.stat().st_size/1024**3:.2f} GiB", flush=True)
        retrieval_root = WORK / "retrieval" / split
        if retrieval_root.exists():
            shutil.rmtree(retrieval_root)
            print(f"Removed completed {split} retrieval intermediates; verified feature Parquets retain the candidate set.")
        if feature_files_ready(split):
            shard_root = WORK / "target_shards" / split
            if shard_root.exists():
                shutil.rmtree(shard_root)
                print(f"Removed completed {split} target shards; source files and feature Parquets remain authoritative.")

    build_feature_split("train", TRAIN_QUERIES)
    del TRAIN_QUERIES

    '''),
    markdown(r"""
    ## 5. Fit XGBoost and LightGBM pair matchers

    Each Source 1 query stays wholly inside one split. XGBoost uses each T4 in turn; LightGBM is a CPU challenger when the Kaggle image already includes it. We select the matcher and threshold using validation macro F0.5, then report that choice on the untouched holdout. No test labels are used.
    """),
    code(r'''
    FEATURES_SQL = ", ".join(f"f.{name}" for name in FEATURES)
    TRUTH_PAIRS = WORK / "truth_pairs.parquet"
    if not TRUTH_PAIRS.is_file():
        db = feature_connection()
        try:
            db.execute(f"""
              COPY (
                SELECT DISTINCT source1_entity_id AS qid, trim(tid) AS tid
                FROM (
                  SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS tid
                  FROM {csv_relation(DATA_ROOT / 'train' / 'train_ground_truth.tsv')}
                  WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
                )
                WHERE trim(tid) <> ''
              ) TO {sql_quote(TRUTH_PAIRS)} (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
        finally:
            db.close()
    ao_parquet_info(TRUTH_PAIRS, ["qid", "tid"])

    MODEL_DIR = WORK / "models"
    MODEL_DIR.mkdir(exist_ok=True)
    MODEL_SCORE_FILES = {"valid": {}, "holdout": {}}

    def labeled_rows(source, cohort):
        feature_path = FEATURE_FILES["train"][source]
        db = feature_connection()
        try:
            labeled = f"""
              SELECT f.qid, f.tid, {FEATURES_SQL},
                     CAST(l.tid IS NOT NULL AS UTINYINT) AS y
              FROM read_parquet({sql_quote(feature_path)}) f
              LEFT JOIN read_parquet({sql_quote(TRUTH_PAIRS)}) l ON f.qid=l.qid AND f.tid=l.tid
              WHERE f.cohort={sql_quote(cohort)}
            """
            if cohort != "fit":
                return db.execute(labeled).df()
            return db.execute(f"""
              WITH labeled AS ({labeled}), ranked_negatives AS (
                SELECT *, row_number() OVER (
                  PARTITION BY qid
                  ORDER BY greatest(coalesce(name_jw, 0), coalesce(address_jw, 0),
                                    coalesce(name_cosine, 0), coalesce(address_cosine, 0)) DESC,
                           tid
                ) AS negative_rank
                FROM labeled WHERE y=0
              )
              SELECT * EXCLUDE (negative_rank) FROM ranked_negatives
              WHERE negative_rank <= {FIT_NEGATIVES_PER_QUERY}
              UNION ALL
              SELECT * FROM labeled WHERE y=1
            """).df()
        finally:
            db.close()

    def model_identity(source):
        return {
            "run_id": RUN_ID, "source": source, "features": FEATURES,
            "train_features_sha256": ao_sha256(FEATURE_FILES["train"][source]),
            "truth_sha256": ao_sha256(TRUTH_PAIRS),
            "xgboost": xgb.__version__, "lightgbm": getattr(lgb, "__version__", None),
            "fit_cohort": "fit", "early_stop_cohort": "valid", "has_lightgbm": HAS_LIGHTGBM,
        }

    def train_source_model(source, gpu):
        marker = MODEL_DIR / f"source{source}.training.complete.json"
        valid_path = MODEL_DIR / f"source{source}_valid_scores.parquet"
        holdout_path = MODEL_DIR / f"source{source}_holdout_scores.parquet"
        expected = model_identity(source)
        if marker.is_file() and valid_path.is_file() and holdout_path.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("identity") == expected:
                MODEL_SCORE_FILES["valid"][source] = valid_path
                MODEL_SCORE_FILES["holdout"][source] = holdout_path
                print(f"Reuse trained source{source} models and validation scores")
                return saved

        train_df = labeled_rows(source, "fit")
        if train_df.empty or train_df.y.nunique() < 2:
            raise RuntimeError(f"Source {source} has insufficient fit rows")
        fit_rows, fit_positives = len(train_df), int(train_df.y.sum())
        fit_sizes = train_df.groupby("qid", sort=False)["qid"].transform("size").to_numpy(dtype=np.float32)
        fit_weights = 1.0 / np.maximum(fit_sizes, 1.0)
        x_train = np.ascontiguousarray(train_df[FEATURES].to_numpy(dtype=np.float32))
        y_train = train_df.y.to_numpy(dtype=np.uint8)
        del train_df, fit_sizes
        gc.collect()

        valid_df = labeled_rows(source, "valid")
        if valid_df.empty or valid_df.y.nunique() < 2:
            raise RuntimeError(f"Source {source} has insufficient validation rows/classes")
        valid_rows = len(valid_df)
        valid_meta = valid_df[["qid", "tid", "y"]]
        valid_sizes = valid_df.groupby("qid", sort=False)["qid"].transform("size").to_numpy(dtype=np.float32)
        valid_weights = 1.0 / np.maximum(valid_sizes, 1.0)
        x_valid = np.ascontiguousarray(valid_df[FEATURES].to_numpy(dtype=np.float32))
        y_valid = valid_df.y.to_numpy(dtype=np.uint8)
        del valid_df, valid_sizes
        gc.collect()

        holdout_df = labeled_rows(source, "holdout")
        holdout_rows = len(holdout_df)
        holdout_meta = holdout_df[["qid", "tid", "y"]]
        x_holdout = np.ascontiguousarray(holdout_df[FEATURES].to_numpy(dtype=np.float32))
        y_holdout = holdout_df.y.to_numpy(dtype=np.uint8)
        del holdout_df
        gc.collect()

        print(f"S{source}: fit={fit_rows:,} pairs ({fit_positives:,} positives; "
              f"up to {FIT_NEGATIVES_PER_QUERY} hard negatives per query); "
              f"valid={valid_rows:,}; holdout={holdout_rows:,}", flush=True)

        dtrain = xgb.DMatrix(x_train, label=y_train, weight=fit_weights, feature_names=FEATURES)
        dvalid = xgb.DMatrix(x_valid, label=y_valid, weight=valid_weights, feature_names=FEATURES)
        xgb_params = {
            **XGBOOST_PARAMS, "device": f"cuda:{gpu}",
            "seed": 2026 + source, "nthread": max(2, THREADS // 2),
        }
        booster = xgb.train(
            xgb_params, dtrain, num_boost_round=XGBOOST_TRAINING["num_boost_round"],
            evals=[(dvalid, "valid")],
            early_stopping_rounds=XGBOOST_TRAINING["early_stopping_rounds"], verbose_eval=100,
        )
        xgb_rounds = int(booster.best_iteration) + 1
        booster.save_model(str(MODEL_DIR / f"source{source}_xgb.json"))
        valid_xgb = booster.predict(dvalid, iteration_range=(0, xgb_rounds)).astype(np.float32)
        del dtrain, dvalid, booster
        gc.collect()
        torch.cuda.empty_cache()

        lgb_model = None
        valid_lgb = np.full(valid_rows, np.nan, dtype=np.float32)
        holdout_lgb = np.full(holdout_rows, np.nan, dtype=np.float32)
        lgb_rounds = 0
        if HAS_LIGHTGBM:
            lgb_params = {
                **LIGHTGBM_PARAMS,
                "n_jobs": max(2, min(8, os.cpu_count() or 4)),
                "random_state": 2026 + source,
            }
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(
                x_train, y_train, sample_weight=fit_weights,
                eval_set=[(x_valid, y_valid)], eval_sample_weight=[valid_weights],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(LIGHTGBM_TRAINING["early_stopping_rounds"], verbose=False),
                    lgb.log_evaluation(100),
                ],
            )
            lgb_rounds = int(lgb_model.best_iteration_ or lgb_model.n_estimators_)
            lgb_model.booster_.save_model(str(MODEL_DIR / f"source{source}_lgb.txt"))
            valid_lgb = lgb_model.predict_proba(x_valid, num_iteration=lgb_rounds)[:, 1].astype(np.float32)
            holdout_lgb = lgb_model.predict_proba(x_holdout, num_iteration=lgb_rounds)[:, 1].astype(np.float32)

        # Save the holdout scores from XGBoost after the model bakeoff; holdout never guides model choice.
        holdout_booster = xgb.Booster()
        holdout_booster.load_model(str(MODEL_DIR / f"source{source}_xgb.json"))
        holdout_booster.set_param({"device": f"cuda:{gpu}", "nthread": max(2, THREADS // 2)})
        dholdout = xgb.DMatrix(x_holdout, feature_names=FEATURES)
        holdout_xgb = holdout_booster.predict(dholdout, iteration_range=(0, xgb_rounds)).astype(np.float32)

        for cohort_df, xgb_score, lgb_score, path in (
            (valid_meta, valid_xgb, valid_lgb, valid_path),
            (holdout_meta, holdout_xgb, holdout_lgb, holdout_path),
        ):
            table = pa.table({
                "qid": cohort_df.qid.to_numpy(), "tid": cohort_df.tid.to_numpy(),
                "y": cohort_df.y.to_numpy(dtype=np.uint8),
                "xgb_score": xgb_score, "lgb_score": lgb_score,
            })
            pq.write_table(table, path, compression="zstd")

        saved = {"identity": expected, "xgb_rounds": xgb_rounds, "lgb_rounds": lgb_rounds,
                 "fit_pairs": fit_rows, "valid_pairs": valid_rows, "holdout_pairs": holdout_rows}
        ao_atomic_json(marker, saved)
        MODEL_SCORE_FILES["valid"][source] = valid_path
        MODEL_SCORE_FILES["holdout"][source] = holdout_path
        del valid_meta, holdout_meta, x_train, y_train, x_valid, y_valid, x_holdout, y_holdout
        del valid_xgb, valid_lgb, holdout_xgb, holdout_lgb, holdout_booster, dholdout, lgb_model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"S{source}: XGBoost={xgb_rounds} rounds; LightGBM={lgb_rounds} rounds", flush=True)
        return saved

    TRAINING_RESULTS = [train_source_model(2, 0), train_source_model(3, 1)]
    '''),
    markdown(r"""
    ## 6. Choose the matcher and threshold on validation; report holdout once

    Candidate links missed by retrieval count as false negatives. Empty-match Source 1 rows remain in the macro score, and the scorer penalizes false positives more heavily through F0.5. The model family, any blend, and threshold are selected only from the `valid` cohort; the `holdout` cohort is evaluated after selection.
    """),
    code(r'''
    def cohort_qids(cohort):
        frame = pd.read_csv(SCOPE_PATH, sep="\t", dtype=str)
        return frame.loc[frame.cohort == cohort, "qid"].tolist()

    VALID_QIDS = cohort_qids("valid")
    HOLDOUT_QIDS = cohort_qids("holdout")

    def truth_counts_for(qids):
        qid_frame = pa.table({"qid": pa.array(qids, type=pa.string())})
        qid_parquet = WORK / "metric_qids.parquet"
        pq.write_table(qid_frame, qid_parquet, compression="zstd")
        db = feature_connection()
        try:
            result = db.execute(f"""
              SELECT q.qid, count(t.tid)::INTEGER AS true_n
              FROM read_parquet({sql_quote(qid_parquet)}) q
              LEFT JOIN read_parquet({sql_quote(TRUTH_PAIRS)}) t USING(qid)
              GROUP BY q.qid ORDER BY q.qid
            """).df()
        finally:
            db.close()
            qid_parquet.unlink(missing_ok=True)
        return result

    def load_scores(paths):
        frames = [pd.read_parquet(path) for path in paths]
        return pd.concat(frames, ignore_index=True)

    VALID_SCORES = load_scores([MODEL_SCORE_FILES["valid"][s] for s in (2, 3)])
    HOLDOUT_SCORES = load_scores([MODEL_SCORE_FILES["holdout"][s] for s in (2, 3)])
    VALID_TRUTH = truth_counts_for(VALID_QIDS)
    HOLDOUT_TRUTH = truth_counts_for(HOLDOUT_QIDS)

    def retrieval_oracle_macro_f05(scores, truth):
        # Perfectly accept every retrieved true pair and reject all retrieved false pairs.
        hits = scores.groupby("qid", as_index=False).y.sum().rename(columns={"y": "retrieved_true_n"})
        joined = truth.merge(hits, on="qid", how="left")
        actual = joined.true_n.to_numpy(dtype=np.float64)
        found = joined.retrieved_true_n.fillna(0).to_numpy(dtype=np.float64)
        denom = 0.25 * actual + found
        per_query = np.ones(len(joined), dtype=np.float64)
        mask = denom > 0
        per_query[mask] = 1.25 * found[mask] / denom[mask]
        return float(per_query.mean())

    VALID_RETRIEVAL_ORACLE = retrieval_oracle_macro_f05(VALID_SCORES, VALID_TRUTH)
    print(f"VALIDATION RETRIEVAL ORACLE (perfect classifier upper bound): {VALID_RETRIEVAL_ORACLE:.6f}")
    HOLDOUT_RETRIEVAL_ORACLE = retrieval_oracle_macro_f05(HOLDOUT_SCORES, HOLDOUT_TRUTH)
    print(f"HOLDOUT RETRIEVAL ORACLE (perfect classifier upper bound): {HOLDOUT_RETRIEVAL_ORACLE:.6f}")
    MODEL_KINDS = ["xgboost"] + (["lightgbm", "blend"] if HAS_LIGHTGBM else [])

    def prepare_metric_arrays(frame, truth):
        qid_codes = pd.Categorical(frame.qid, categories=truth.qid).codes.astype(np.int32)
        if (qid_codes < 0).any():
            raise RuntimeError("Scored candidate has a qid outside the metric cohort")
        xgb_scores = frame.xgb_score.to_numpy(dtype=np.float32)
        scores = {"xgboost": xgb_scores}
        if HAS_LIGHTGBM:
            lgb_scores = frame.lgb_score.to_numpy(dtype=np.float32)
            scores["lightgbm"] = lgb_scores
            scores["blend"] = ((xgb_scores + lgb_scores) / 2).astype(np.float32)
        return {
            "qid_codes": qid_codes,
            "labels": frame.y.to_numpy(dtype=np.uint8),
            "scores": scores,
            "true_counts": truth.true_n.to_numpy(dtype=np.float64),
            "query_count": len(truth),
        }

    VALID_METRICS = prepare_metric_arrays(VALID_SCORES, VALID_TRUTH)
    HOLDOUT_METRICS = prepare_metric_arrays(HOLDOUT_SCORES, HOLDOUT_TRUTH)

    def macro_f05_at(metric, kind, threshold):
        scores = metric["scores"][kind]
        qid_codes = metric["qid_codes"]
        keep = scores >= threshold
        predicted = np.bincount(qid_codes[keep], minlength=metric["query_count"])
        hits = np.bincount(
            qid_codes[keep & (metric["labels"] == 1)], minlength=metric["query_count"]
        )
        actual = metric["true_counts"]
        denom = 0.25 * actual + predicted
        per_query = np.ones(metric["query_count"], dtype=np.float64)
        nonzero = denom > 0
        per_query[nonzero] = 1.25 * hits[nonzero] / denom[nonzero]
        return float(per_query.mean())

    threshold_grid = np.unique(np.r_[
        [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03],
        np.arange(0.05, 0.951, 0.05), [0.975, 0.99, 0.995, 0.999],
    ])
    validation_rows = []
    for kind in MODEL_KINDS:
        coarse = [{"model": kind, "threshold": float(t),
                   "macro_f05": macro_f05_at(VALID_METRICS, kind, float(t))}
                  for t in threshold_grid]
        best_coarse = max(coarse, key=lambda row: (row["macro_f05"], row["threshold"]))
        fine_grid = np.arange(max(0.005, best_coarse["threshold"] - 0.045),
                              min(0.999, best_coarse["threshold"] + 0.046), 0.005)
        validation_rows.extend(coarse)
        validation_rows.extend({"model": kind, "threshold": float(t),
                                "macro_f05": macro_f05_at(VALID_METRICS, kind, float(t))}
                               for t in fine_grid)

    priority = {"xgboost": 0, "lightgbm": 1, "blend": 2}
    BEST = max(validation_rows, key=lambda row: (row["macro_f05"], row["threshold"], -priority[row["model"]]))
    BEST["validation_true_links"] = int(VALID_SCORES.y.sum())
    BEST["validation_candidate_link_recall"] = float(VALID_SCORES.y.sum() / max(1, VALID_TRUTH.true_n.sum()))
    BEST["validation_retrieval_oracle_macro_f05"] = VALID_RETRIEVAL_ORACLE
    BEST["validation_source1_rows"] = len(VALID_TRUTH)
    ao_atomic_json(WORK / "validation_summary.json", {"best": BEST, "all_thresholds": validation_rows})

    HOLDOUT = {
        "model": BEST["model"], "threshold": BEST["threshold"],
        "macro_f05": macro_f05_at(HOLDOUT_METRICS, BEST["model"], BEST["threshold"]),
        "candidate_link_recall": float(HOLDOUT_SCORES.y.sum() / max(1, HOLDOUT_TRUTH.true_n.sum())),
        "retrieval_oracle_macro_f05": HOLDOUT_RETRIEVAL_ORACLE,
        "source1_rows": len(HOLDOUT_TRUTH),
        "true_links": int(HOLDOUT_TRUTH.true_n.sum()),
    }
    ao_atomic_json(WORK / "holdout_summary.json", HOLDOUT)
    print("VALIDATION (used for model/threshold selection):", BEST)
    if min(VALID_RETRIEVAL_ORACLE, HOLDOUT_RETRIEVAL_ORACLE) < TARGET_MACRO_F05:
        print(f"WARNING: at least one clean cohort's retrieval oracle is below {TARGET_MACRO_F05:.1%}; "
              "the matcher cannot close that gap without better retrieval.")
    print("UNTOUCHED HOLDOUT (reported once after selection):", HOLDOUT)
    if BEST["validation_candidate_link_recall"] < 0.90:
        print("WARNING: candidate recall is below 90%; inspect the retrieval misses before trusting final scores.")
    del VALID_METRICS, HOLDOUT_METRICS
    gc.collect()
    '''),
    markdown(r"""
    ## 7. Score every test candidate and assemble the required files

    Test candidate generation is intentionally behind a measurable quality gate. If either clean cohort's perfect-classifier retrieval oracle falls below the 98.9% target, the notebook stops before the much larger test retrieval; set `FORCE_TEST_RUN_BELOW_TARGET = True` in setup only if you still want a lower-ceiling submission. Inference is chunked and writes a completed checkpoint only after every candidate row in that source file is scored. Assembly checks exact test Source 1 coverage, known target IDs, unique candidate pairs, the 128-candidate cap, and that every predicted match is among the submitted candidates.
    """),
    code(r"""
    if (min(VALID_RETRIEVAL_ORACLE, HOLDOUT_RETRIEVAL_ORACLE) < TARGET_MACRO_F05
            and not FORCE_TEST_RUN_BELOW_TARGET):
        raise RuntimeError(
            f"Quality gate stopped before test retrieval: clean-cohort retrieval oracle is "
            f"{min(VALID_RETRIEVAL_ORACLE, HOLDOUT_RETRIEVAL_ORACLE):.4%}, below the "
            f"{TARGET_MACRO_F05:.1%} target. Improve retrieval first, or set "
            "FORCE_TEST_RUN_BELOW_TARGET=True to proceed with a lower-ceiling run."
        )

    build_feature_split("test")

    def selected_model_pointer(source):
        pointer = MODEL_DIR / f"source{source}.json"
        kind = BEST["model"]
        content = {
            "selected_model": kind, "threshold": BEST["threshold"],
            "source": source, "run_id": RUN_ID,
            "xgboost_file": f"source{source}_xgb.json",
            "xgboost_sha256": ao_sha256(MODEL_DIR / f"source{source}_xgb.json"),
            "xgboost_rounds": int(json.loads((MODEL_DIR / f"source{source}.training.complete.json").read_text())["xgb_rounds"]),
            "feature_file_sha256": ao_sha256(FEATURE_FILES["test"][source]),
        }
        if kind in {"lightgbm", "blend"}:
            lgb_path = MODEL_DIR / f"source{source}_lgb.txt"
            content.update({"lightgbm_file": lgb_path.name, "lightgbm_sha256": ao_sha256(lgb_path)})
        ao_atomic_json(pointer, content)
        return pointer, content

    def infer_source(source, gpu):
        part = FEATURE_FILES["test"][source]
        out = WORK / "accepted" / f"source{source}_all.parquet"
        partial = out.with_name(out.stem + ".partial.parquet")
        marker = out.with_name(out.name + ".complete.json")
        pointer, pointer_data = selected_model_pointer(source)
        candidate_identity = ao_fingerprint(part)
        candidate_hash = ao_sha256(part)
        existing = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        if (out.is_file() and existing.get("candidate_sha256") == candidate_hash
                and existing.get("model_sha256") == ao_sha256(pointer)
                and existing.get("threshold") == BEST["threshold"]
                and ao_parquet_info(out, ["qid", "tid"]) == existing.get("accepted_rows")):
            print(f"Reuse completed inference: {out.name}")
            return out
        marker.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        xgb_model = None
        lgb_model = None
        kind = BEST["model"]
        if kind in {"xgboost", "blend"}:
            if ao_sha256(MODEL_DIR / f"source{source}_xgb.json") != pointer_data["xgboost_sha256"]:
                raise RuntimeError(f"XGBoost model changed after selection for Source {source}")
            xgb_model = xgb.Booster()
            xgb_model.load_model(str(MODEL_DIR / f"source{source}_xgb.json"))
            xgb_model.set_param({"device": f"cuda:{gpu}", "nthread": max(2, THREADS // 2)})
        if kind in {"lightgbm", "blend"}:
            lgb_model = lgb.Booster(model_file=str(MODEL_DIR / f"source{source}_lgb.txt"))
            if ao_sha256(MODEL_DIR / f"source{source}_lgb.txt") != pointer_data["lightgbm_sha256"]:
                raise RuntimeError(f"LightGBM model changed after selection for Source {source}")
        train_record = json.loads((MODEL_DIR / f"source{source}.training.complete.json").read_text(encoding="utf-8"))
        xgb_rounds = int(train_record["xgb_rounds"])
        lgb_rounds = int(train_record["lgb_rounds"])
        prediction_rounds = max(xgb_rounds if xgb_model else 0, lgb_rounds if lgb_model else 0, 1)
        accepted = scored = 0
        with pq.ParquetFile(part) as reader, pq.ParquetWriter(
            partial, pa.schema([pa.field("qid", pa.string()), pa.field("tid", pa.string())]), compression="zstd"
        ) as writer:
            expected_rows = reader.metadata.num_rows
            for batch in reader.iter_batches(batch_size=200_000, columns=["qid", "tid"] + FEATURES):
                frame = batch.to_pandas()
                matrix = np.ascontiguousarray(frame[FEATURES].to_numpy(dtype=np.float32))
                scores = []
                if xgb_model is not None:
                    dmatrix = xgb.DMatrix(matrix, feature_names=FEATURES)
                    scores.append(xgb_model.predict(dmatrix, iteration_range=(0, xgb_rounds)).astype(np.float32))
                    del dmatrix
                if lgb_model is not None:
                    scores.append(lgb_model.predict(matrix, num_iteration=lgb_rounds).astype(np.float32))
                score = scores[0] if len(scores) == 1 else ((scores[0] + scores[1]) / 2).astype(np.float32)
                keep = score >= BEST["threshold"]
                scored += len(frame)
                if keep.any():
                    selected = frame.loc[keep, ["qid", "tid"]]
                    writer.write_table(pa.Table.from_pandas(selected, preserve_index=False))
                    accepted += len(selected)
                del frame, matrix, scores, score, batch
                if scored % 2_000_000 < 200_000:
                    print(f"S{source}: scored {scored:,}/{expected_rows:,}; accepted={accepted:,}", flush=True)
        if scored != expected_rows or ao_fingerprint(part) != candidate_identity:
            raise RuntimeError(f"Incomplete inference or candidate file changed: {part}")
        if (xgb_model is not None and ao_sha256(MODEL_DIR / f"source{source}_xgb.json") != pointer_data["xgboost_sha256"]):
            raise RuntimeError(f"XGBoost model changed during inference for Source {source}")
        if (lgb_model is not None and ao_sha256(MODEL_DIR / f"source{source}_lgb.txt") != pointer_data["lightgbm_sha256"]):
            raise RuntimeError(f"LightGBM model changed during inference for Source {source}")
        os.replace(partial, out)
        ao_atomic_json(marker, {
            "candidate": candidate_identity, "candidate_sha256": candidate_hash,
            "scored_rows": scored, "accepted_rows": accepted,
            "accepted_sha256": ao_sha256(out), "model_file": pointer.name,
            "model_sha256": ao_sha256(pointer), "threshold": BEST["threshold"],
            "prediction_rounds": prediction_rounds,
        })
        del xgb_model, lgb_model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Inference S{source}: {scored:,} scored, {accepted:,} accepted")
        return out

    ACCEPTED = [infer_source(2, 0), infer_source(3, 1)]
    CANDIDATE_PARTS = [FEATURE_FILES["test"][2], FEATURE_FILES["test"][3]]
    RESULT = assemble_outputs(
        CANDIDATE_PARTS, ACCEPTED, DATA_ROOT, WORK,
        max_candidates_per_entity=SETTINGS.cap,
    )

    validator = DATA_ROOT.parent / "utils" / "validate_submission.py"
    if validator.is_file():
        check = subprocess.run([
            sys.executable, str(validator),
            "--matching", str(OUTPUT / "matching_results.tsv"),
            "--candidate", str(OUTPUT / "candidate_pairs.tsv"),
            "--test-dir", str(DATA_ROOT / "test"),
        ], cwd=DATA_ROOT.parent, capture_output=True, text=True)
        if check.stdout:
            print(check.stdout, end="")
        if check.stderr:
            print(check.stderr, end="", file=sys.stderr)
        if check.returncode:
            raise RuntimeError(f"Official submission validator failed with exit code {check.returncode}")
        completion_path = WORK / "output_completion.json"
        completion_record = json.loads(completion_path.read_text(encoding="utf-8"))
        completion_record["official_validator_run"] = True
        ao_atomic_json(completion_path, completion_record)
        RESULT["official_validator_run"] = True
    else:
        print("Official validator script not attached; internal output checks passed, but external format validation was not run.")

    manifest = {
        "run_id": RUN_ID, "pipeline_version": PIPELINE_VERSION,
        "build_code_sha256": BUILD_CODE_SHA256,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "model_configuration": MODEL_CONFIGURATION,
        "runtime_versions": RUNTIME_VERSIONS,
        "input_files": INPUT_IDS, "retrieval_commit": "09bb350a7dd52393cf8d27e002c8ffdf67840685",
        "retrieval_settings": vars(SETTINGS), "query_policy": QUERY_POLICY,
        "training_policy": TRAINING_POLICY,
        "train_query_scope_rows": TRAIN_QUERY_COUNT,
        "validation": BEST, "holdout": HOLDOUT,
        "xgboost": xgb.__version__, "lightgbm": getattr(lgb, "__version__", None),
        "duckdb": duckdb.__version__, "torch": torch.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
        "output_validation": RESULT["validation"],
        "official_validator_run": RESULT["official_validator_run"],
        "output_files": RESULT["outputs"],
    }
    ao_atomic_json(WORK / "run_manifest.json", manifest)
    print("RUN COMPLETE:", json.dumps({"validation": BEST, "holdout": HOLDOUT,
          "output_validation": RESULT["validation"], "outputs": RESULT["outputs"]}, indent=2))
    print(f"matching_results.tsv: {OUTPUT / 'matching_results.tsv'}")
    print(f"candidate_pairs.tsv:   {OUTPUT / 'candidate_pairs.tsv'}")
    """),
    markdown(r"""
    ## License and attribution

    The candidate retrieval modules embedded above come from the team branch named in the setup section, with the gzip reader adaptation documented there. They are distributed under Apache License 2.0. Copyright and license terms follow.

    ```text
    """ + license_text.rstrip() + r"""
    ```
    """),
]

for cell in cells:
    if cell.cell_type == "code":
        cell.source = cell.source.replace("BUILD_CODE_SHA256_PLACEHOLDER", BUILD_CODE_SHA256)

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "kaggle": {"accelerator": "GPU T4 x2", "internet": True},
}
nbf.validate(notebook)
nbf.write(notebook, OUT)
print(f"Wrote {OUT} with {len(cells)} cells; embedded candidate sources from {SOURCE_REF}.")
