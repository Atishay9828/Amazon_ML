"""Memory-bounded submission assembly and recovery from completed Parquet files.

This module is embedded into both notebooks by sync_notebook.py. It imports no
GPU framework and never trains a model or changes the accepted candidate set.
"""

import gc
import hashlib
import heapq
import itertools
import json
import os
import shutil
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


ASSEMBLY_VERSION = "streaming-v1"


def ao_log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def ao_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def ao_file_list(paths):
    return "[" + ",".join(ao_quote(p) for p in paths) + "]"


def ao_csv(path):
    return f"read_csv({ao_quote(path)}, delim='\\t', header=true, all_varchar=true)"


def ao_resources(work):
    free_gb = shutil.disk_usage(work).free / 1024**3
    rss = "unavailable"
    try:
        import psutil
        rss = f"{psutil.Process().memory_info().rss / 1024**3:.2f} GiB"
    except ImportError:
        pass
    ao_log(f"Resources: process RAM={rss}; disk free={free_gb:.2f} GiB")


def ao_connection(work, memory_mb=768):
    # Only external sorts and ordinary joins/counts run here. No list/string
    # aggregate, dataframe materialization, or Python-wide ID set is used.
    tmp = work / "assembly_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free
    if free < 512 * 1024**2:
        raise RuntimeError("Less than 512 MiB disk free. Preserve the original Parquets and free scratch space before recovery.")
    db = duckdb.connect()
    try:
        db.execute(f"SET memory_limit = '{int(memory_mb)}MB'")
        db.execute("SET threads = 1")
        db.execute("SET preserve_insertion_order = true")
        db.execute(f"SET temp_directory = {ao_quote(tmp)}")
        spill_bytes = min(4 * 1024**3, max(256 * 1024**2, free // 2))
        db.execute(f"SET max_temp_directory_size = '{spill_bytes}B'")
        return db
    except Exception:
        db.close()
        raise


def ao_atomic_json(path, value):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(partial, path)


def ao_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def ao_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def ao_parquet_info(path, required):
    if not Path(path).is_file():
        raise RuntimeError(f"Required saved file is missing: {path}. Recovery cannot recreate missing predictions.")
    try:
        with pq.ParquetFile(path) as reader:
            missing = set(required) - set(reader.schema_arrow.names)
            if missing:
                raise ValueError(f"Missing columns: {sorted(missing)}")
            return reader.metadata.num_rows
    except Exception as error:
        raise RuntimeError(f"Saved Parquet is incomplete or invalid: {path}. Preserve the other files; this partition needs regeneration.") from error


def ao_sorted_file(path, work, pair_file=True):
    path = Path(path)
    identity = {"assembly_version": ASSEMBLY_VERSION, "pair_file": pair_file,
                "input": ao_fingerprint(path)}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache = work / "assembly_sorted"
    cache.mkdir(exist_ok=True)
    out = cache / f"{digest}.parquet"
    marker = cache / f"{digest}.json"
    columns = ["qid", "tid"] if pair_file else ["qid"]
    expected_rows = ao_parquet_info(path, columns) if pair_file else None
    if out.exists() and marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        actual_rows = ao_parquet_info(out, columns)
        if saved.get("identity") == identity and actual_rows == saved.get("rows"):
            if expected_rows is None or actual_rows == expected_rows:
                ao_log(f"Reuse sorted IDs: {path.name} ({actual_rows:,} rows)")
                return out, actual_rows
    ao_resources(work)
    ao_log(f"External sort with one CPU thread and 768 MB DuckDB budget: {path.name}")
    partial = out.with_name(out.stem + ".partial.parquet")
    db = ao_connection(work)
    try:
        if pair_file:
            query = f"SELECT qid, tid FROM read_parquet({ao_quote(path)}) ORDER BY qid, tid"
        else:
            expected_rows = db.execute(f"SELECT count(*) FROM {ao_csv(path)}").fetchone()[0]
            query = f"SELECT entity_id AS qid FROM {ao_csv(path)} ORDER BY entity_id"
        db.execute(f"COPY ({query}) TO {ao_quote(partial)} (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 65536)")
    finally:
        db.close()
        gc.collect()
    actual_rows = ao_parquet_info(partial, columns)
    if actual_rows != expected_rows:
        raise RuntimeError(f"Sorted row count changed for {path}: {expected_rows} -> {actual_rows}")
    os.replace(partial, out)
    ao_atomic_json(marker, {"identity": identity, "rows": actual_rows})
    return out, actual_rows


def ao_rows(path, columns):
    previous = None
    with pq.ParquetFile(path) as reader:
        for batch in reader.iter_batches(batch_size=8192, columns=columns, use_threads=False):
            values = [batch.column(i).to_pylist() for i in range(len(columns))]
            for row in zip(*values):
                if any(value is None or not isinstance(value, str) for value in row):
                    raise RuntimeError(f"Null or non-string identifier in {path}")
                if previous is not None and row < previous:
                    raise RuntimeError(f"Sorted cache is out of order: {path}")
                previous = row
                yield row


def ao_grouped_pairs(sorted_paths, maximum):
    streams = [ao_rows(path, ["qid", "tid"]) for path in sorted_paths]
    try:
        merged = heapq.merge(*streams)
        for qid, rows in itertools.groupby(merged, key=lambda row: row[0]):
            ids = []
            previous = None
            for _, tid in rows:
                if tid == previous:
                    raise RuntimeError(f"Duplicate pair encountered for {qid}; no pairs were silently removed.")
                if not tid.startswith(("S2-", "S3-")) or any(c in tid for c in ",\t\r\n"):
                    raise RuntimeError(f"Invalid target ID for {qid}: {tid!r}")
                ids.append(tid)
                if len(ids) > maximum:
                    raise RuntimeError(f"{qid} exceeds the configured {maximum}-candidate bound. Check the selected run files.")
                previous = tid
            yield qid, ids
    finally:
        for stream in streams:
            stream.close()


def ao_check_target_ids(candidate_parts, test_dir, work):
    for source, prefix in (("source2", "S2-"), ("source3", "S3-")):
        parts = [p for p in candidate_parts if p.name.startswith(source + "_")]
        if not parts:
            raise RuntimeError(f"No {source} candidate partitions were selected")
        target_path = test_dir / f"test_{source}.tsv"
        if not target_path.is_file():
            raise RuntimeError(f"Missing official target source: {target_path}")
        ao_log(f"Check candidate IDs against the official test {source} file")
        db = ao_connection(work)
        try:
            bad_prefix = db.execute(f"SELECT count(*) FROM read_parquet({ao_file_list(parts)}) WHERE tid IS NULL OR NOT starts_with(tid, {ao_quote(prefix)})").fetchone()[0]
            invalid = db.execute(f"""
              SELECT count(*) FROM read_parquet({ao_file_list(parts)}) p
              ANTI JOIN {ao_csv(target_path)} t ON p.tid = t.entity_id
            """).fetchone()[0]
            if bad_prefix or invalid:
                raise RuntimeError(f"{source}: wrong-prefix rows={bad_prefix}, unknown-target rows={invalid}")
        finally:
            db.close()
            gc.collect()


def assemble_outputs(candidate_parts, accepted_parts, data_root, work_root,
                     max_candidates_per_entity=80, expected_accepted_counts=None):
    """Sort narrow pair files sequentially; merge/write one Source 1 at a time."""
    data_root, work = Path(data_root), Path(work_root)
    candidate_parts = [Path(p) for p in candidate_parts]
    accepted_parts = [Path(p) for p in accepted_parts]
    if not candidate_parts or len(candidate_parts) != len(accepted_parts):
        raise RuntimeError("Need a complete candidate/accepted partition pair for every source and country")
    if len({p.name for p in candidate_parts}) != len(candidate_parts):
        raise RuntimeError("Duplicate candidate partition names")
    if sorted(p.name for p in candidate_parts) != sorted(p.name for p in accepted_parts):
        raise RuntimeError("Candidate and accepted partition filenames do not describe the same run")
    candidate_count = sum(ao_parquet_info(p, ["qid", "tid"]) for p in candidate_parts)
    accepted_count = sum(ao_parquet_info(p, ["qid", "tid"]) for p in accepted_parts)
    expected_accepted_counts = expected_accepted_counts or {}
    candidate_by_name = {p.name: p for p in candidate_parts}
    completion_evidence = {}
    sidecar_flags = [p.with_name(p.name + ".complete.json").exists() for p in accepted_parts]
    if any(sidecar_flags) and not all(sidecar_flags):
        raise RuntimeError("Mixed old and new inference checkpoints. Keep separate runs together; do not assemble a mixture.")
    saved_thresholds = set()
    source_models = {}
    for part in accepted_parts:
        count = ao_parquet_info(part, ["qid", "tid"])
        sidecar = part.with_name(part.name + ".complete.json")
        candidate = candidate_by_name[part.name]
        if sidecar.exists():
            evidence = json.loads(sidecar.read_text(encoding="utf-8"))
            expected_rows = ao_parquet_info(candidate, ["qid", "tid"])
            # Content identity survives restoring the run under a different path.
            if (evidence.get("candidate_sha256") != ao_sha256(candidate)
                    or evidence.get("accepted_rows") != count
                    or evidence.get("scored_rows") != expected_rows):
                raise RuntimeError(f"Inference completion record does not match {part.name}")
            source = part.name.split("_", 1)[0]
            if evidence.get("model_file") != source + ".json":
                raise RuntimeError(f"Wrong source model in checkpoint: {part.name}")
            model_path = work / "models" / evidence["model_file"]
            if not model_path.is_file() or ao_sha256(model_path) != evidence["model_sha256"]:
                raise RuntimeError(f"Saved model changed after inference for {part.name}")
            if ao_sha256(part) != evidence.get("accepted_sha256"):
                raise RuntimeError(f"Accepted file changed after inference for {part.name}")
            saved_threshold = evidence.get("threshold")
            rounds = evidence.get("prediction_rounds")
            if not isinstance(saved_threshold, (int, float)) or not 0 <= saved_threshold <= 1:
                raise RuntimeError(f"Invalid threshold in checkpoint: {part.name}")
            if not isinstance(rounds, int) or rounds < 1:
                raise RuntimeError(f"Invalid prediction rounds in checkpoint: {part.name}")
            identity = (evidence["model_sha256"], rounds)
            if source_models.setdefault(source, identity) != identity:
                raise RuntimeError(f"Different models or prediction rounds for {source}")
            saved_thresholds.add(saved_threshold)
            completion_evidence[part.name] = evidence
        elif part.name in expected_accepted_counts:
            expected = expected_accepted_counts[part.name]
            if count != expected:
                raise RuntimeError(f"{part.name}: {count:,} rows, but completed inference log reported {expected:,}; do not use this partial file")
            completion_evidence[part.name] = {
                "accepted_rows": count, "evidence": "matched AJ's supplied completed-inference log",
                "model_and_threshold_hashes": "not available for this legacy checkpoint",
            }
        else:
            raise RuntimeError(f"No completion record for {part.name}. Supply its accepted count from a completed inference log; a valid footer alone is insufficient.")
    if len(saved_thresholds) > 1:
        raise RuntimeError("Inference checkpoints use different thresholds; do not combine runs")
    ao_log(f"ASSEMBLY START: {candidate_count:,} candidate pairs; {accepted_count:,} accepted pairs")
    ao_resources(work)
    source_files = [data_root / "test" / f"test_source{i}.tsv" for i in (1, 2, 3)]
    input_files = candidate_parts + accepted_parts + source_files
    snapshots = [ao_fingerprint(p) for p in input_files]
    # An interrupted rerun must not leave a stale success marker for new files.
    (work / "output_completion.json").unlink(missing_ok=True)
    ao_check_target_ids(candidate_parts, data_root / "test", work)
    source1 = data_root / "test" / "test_source1.tsv"
    sorted_s1, expected_s1 = ao_sorted_file(source1, work, pair_file=False)
    sorted_candidates = [ao_sorted_file(p, work)[0] for p in candidate_parts]
    sorted_accepted = [ao_sorted_file(p, work)[0] for p in accepted_parts]

    output = work / "output"
    output.mkdir(exist_ok=True)
    candidate_final = output / "candidate_pairs.tsv"
    matching_final = output / "matching_results.tsv"
    candidate_partial = output / "candidate_pairs.tsv.partial"
    matching_partial = output / "matching_results.tsv.partial"
    chash, mhash = hashlib.sha256(), hashlib.sha256()
    cgroups = ao_grouped_pairs(sorted_candidates, max_candidates_per_entity)
    mgroups = ao_grouped_pairs(sorted_accepted, max_candidates_per_entity)
    source_rows = ao_rows(sorted_s1, ["qid"])
    cgroup, mgroup = next(cgroups, None), next(mgroups, None)
    rows = seen_candidates = seen_matches = empty_candidates = 0
    previous_qid = None
    ao_log("Stream both TSV files; validate coverage, duplicates and match/candidate subsets")
    try:
        with candidate_partial.open("wb", buffering=1024**2) as cf, matching_partial.open("wb", buffering=1024**2) as mf:
            cheader = b"source1_entity_id\tcandidate_entity_ids\n"
            mheader = b"source1_entity_id\tmatched_entity_ids\n"
            cf.write(cheader); chash.update(cheader)
            mf.write(mheader); mhash.update(mheader)
            for (qid,) in source_rows:
                if qid == previous_qid or not qid.startswith("S1-") or any(c in qid for c in ",\t\r\n"):
                    raise RuntimeError("Invalid or duplicated Source 1 ID in official input")
                if cgroup is not None and cgroup[0] < qid:
                    raise RuntimeError(f"Candidate Source 1 ID absent from test: {cgroup[0]}")
                if mgroup is not None and mgroup[0] < qid:
                    raise RuntimeError(f"Accepted Source 1 ID absent from test: {mgroup[0]}")
                cids, mids = [], []
                if cgroup is not None and cgroup[0] == qid:
                    cids = cgroup[1]
                    cgroup = next(cgroups, None)
                if mgroup is not None and mgroup[0] == qid:
                    mids = mgroup[1]
                    mgroup = next(mgroups, None)
                if not set(mids).issubset(cids):
                    raise RuntimeError(f"Accepted ID missing from the scored candidate set for {qid}")
                cline = (qid + "\t" + ",".join(cids) + "\n").encode("utf-8")
                mline = (qid + "\t" + ",".join(mids) + "\n").encode("utf-8")
                cf.write(cline); chash.update(cline)
                mf.write(mline); mhash.update(mline)
                rows += 1
                seen_candidates += len(cids)
                seen_matches += len(mids)
                empty_candidates += not cids
                previous_qid = qid
                if rows % 100_000 == 0:
                    ao_log(f"  assembled {rows:,}/{expected_s1:,} Source 1 entities")
            if cgroup is not None or mgroup is not None:
                raise RuntimeError("Pairs remain for Source 1 IDs absent from the official test source")
            if (rows, seen_candidates, seen_matches) != (expected_s1, candidate_count, accepted_count):
                raise RuntimeError("Row totals differ from the complete input Parquet/source counts")
            cf.flush(); os.fsync(cf.fileno())
            mf.flush(); os.fsync(mf.fileno())
    finally:
        source_rows.close(); cgroups.close(); mgroups.close()
    if snapshots != [ao_fingerprint(p) for p in input_files]:
        raise RuntimeError("Input files changed during assembly; no completion marker was written")
    os.replace(candidate_partial, candidate_final)
    os.replace(matching_partial, matching_final)
    result = {
        "assembly_version": ASSEMBLY_VERSION, "source1_rows": rows,
        "candidate_pairs": seen_candidates, "accepted_pairs": seen_matches,
        "source1_without_candidates": empty_candidates,
        "validation": "PASS: exact Source 1 coverage, target existence, unique pairs and match subsets",
        "official_validator_run": False,
        "inference_completion_evidence": completion_evidence,
        "input_files": snapshots,
        "outputs": {
            candidate_final.name: {"bytes": candidate_final.stat().st_size, "sha256": chash.hexdigest()},
            matching_final.name: {"bytes": matching_final.stat().st_size, "sha256": mhash.hexdigest()},
        },
    }
    ao_atomic_json(work / "output_completion.json", result)
    ao_log(f"OUTPUT COMPLETE: {rows:,} Source 1 rows, {seen_matches:,} final links. Files: {output}")
    ao_resources(work)
    return result


def recover_saved_outputs(data_root=None, work_root=None,
                          candidate_tag="v3_64_512_3_40", expected_accepted_counts=None):
    work = Path(work_root or "/kaggle/working/aj_entity_resolution")
    if not work.is_dir():
        raise RuntimeError(f"Saved run directory is missing: {work}. Restore its Parquets before recovery.")
    if data_root is None:
        roots = sorted({p.parent.parent.resolve() for p in Path("/kaggle/input").glob("**/dataset/test/test_source1.tsv")})
        if len(roots) != 1:
            raise RuntimeError(f"Expected one attached official dataset; found {roots}. Set data_root explicitly.")
        data_root = roots[0]
    data_root = Path(data_root)
    db = ao_connection(work)
    try:
        countries = [row[0] for row in db.execute(f"SELECT DISTINCT country FROM {ao_csv(data_root / 'test' / 'test_source1.tsv')} ORDER BY country").fetchall()]
    finally:
        db.close()
    candidates, accepted = [], []
    for country in countries:
        country_key = hashlib.sha1(country.encode()).hexdigest()[:10]
        for source in ("source2", "source3"):
            filename = f"{source}_{country_key}_{candidate_tag}.parquet"
            candidates.append(work / "test_candidates" / filename)
            accepted.append(work / "accepted" / filename)
    ao_log("RECOVERY ONLY: reuse existing Parquets; no training, inference or GPU imports")
    return assemble_outputs(candidates, accepted, data_root, work,
                            expected_accepted_counts=expected_accepted_counts)
