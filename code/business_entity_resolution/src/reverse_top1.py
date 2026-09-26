"""Map each Source 2/3 record to its nearest Source 1 records in the same split.

The default pass keeps the top-1 Source 1 record for every target on name and
address. The blank-address pass keeps the top-k Source 1 records on the name
alone, only for targets without an address, where the joint text is diluted.

The output is a proposal channel for candidates.py, not a predicted match.
No ground-truth labels or external business data are read.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import time
from itertools import islice
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

try:
    from .data import iter_records, sha256_file, source_path
    from .normalization import normalize_address, normalize_name
except ImportError:
    from data import iter_records, sha256_file, source_path
    from normalization import normalize_address, normalize_name


HEADER = ("source1_entity_id", "candidate_entity_id")
_STATE = None


def _text(record) -> str:
    return normalize_name(record.business_name) + " | " + normalize_address(record.business_address)


def _name_text(record) -> str:
    return normalize_name(record.business_name)


def islice_batches(records, size: int, text=_text):
    iterator = iter(records)
    while batch := list(islice(iterator, size)):
        yield [(record.entity_id, text(record)) for record in batch]


def _batches(records, size: int):
    return islice_batches(records, size, _text)


def _top1_batch(batch: list[tuple[str, str]]) -> tuple[int, list[tuple[str, str]]]:
    state = _STATE
    if state is None:
        raise RuntimeError("reverse index not initialized before worker fork")
    source1_ids, vectorizer, keep, tfidf, index_t = state
    queries = vectorizer.transform([text for _, text in batch]).tocsr()
    queries.data[~keep[queries.indices]] = 0
    queries.eliminate_zeros()
    scores = (tfidf.transform(queries).astype(np.float32) @ index_t).tocsr()
    pairs = []
    for row, (target_id, _) in enumerate(batch):
        start, stop = scores.indptr[row:row + 2]
        if start == stop:
            continue
        values = scores.data[start:stop]
        columns = scores.indices[start:stop]
        best = int(columns[values == values.max()].min())
        pairs.append((source1_ids[best], target_id))
    return len(batch), pairs


def _topk_batch(batch: list[tuple[str, str]]) -> tuple[int, list[tuple[str, str]]]:
    state = _STATE
    if state is None:
        raise RuntimeError("reverse index not initialized before worker fork")
    source1_ids, vectorizer, keep, tfidf, index_t, k = state
    queries = vectorizer.transform([text for _, text in batch]).tocsr()
    queries.data[~keep[queries.indices]] = 0
    queries.eliminate_zeros()
    scores = (tfidf.transform(queries).astype(np.float32) @ index_t).tocsr()
    pairs = []
    for row, (target_id, _) in enumerate(batch):
        start, stop = scores.indptr[row:row + 2]
        values = scores.data[start:stop]
        columns = scores.indices[start:stop]
        if len(values) > k:
            chosen = np.argpartition(-values, k - 1)[:k]
            values, columns = values[chosen], columns[chosen]
        for column in columns[np.lexsort((columns, -values))]:
            pairs.append((source1_ids[int(column)], target_id))
    return len(batch), pairs


def build_index(data_root: Path, split: str, hash_features: int,
                df_cap_fraction: float, matrix_chunk: int, text=_text):
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(3, 3),
                                   n_features=hash_features, alternate_sign=False,
                                   norm=None, dtype=np.float32)
    source1_ids = []
    blocks = []
    for batch in islice_batches(iter_records(source_path(data_root, split, 1), 1), matrix_chunk, text):
        source1_ids.extend(record_id for record_id, _ in batch)
        blocks.append(vectorizer.transform([value for _, value in batch]))
    if not blocks:
        raise ValueError("Source 1 has no records")
    matrix = sparse.vstack(blocks, format="csr", dtype=np.float32)
    del blocks
    df = np.bincount(matrix.indices, minlength=hash_features)
    keep = df <= max(50, int(df_cap_fraction * len(source1_ids)))
    matrix.data[~keep[matrix.indices]] = 0
    matrix.eliminate_zeros()
    tfidf = TfidfTransformer(norm="l2", sublinear_tf=True).fit(matrix)
    index_t = tfidf.transform(matrix).astype(np.float32).T.tocsr()
    return source1_ids, vectorizer, keep, tfidf, index_t


def run(data_root: Path, split: str, out: Path, *, workers: int = 4,
        batch_size: int = 256, hash_features: int = 1 << 22,
        df_cap_fraction: float = 0.002, matrix_chunk: int = 200_000,
        limit_targets: int | None = None, config_hash: str | None = None) -> dict:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if workers < 1 or batch_size < 1 or matrix_chunk < 1 or hash_features < 2:
        raise ValueError("worker, batch, matrix, and feature counts must be positive")
    if not 0 < df_cap_fraction <= 1:
        raise ValueError("df_cap_fraction must be in (0, 1]")
    started = time.monotonic()
    global _STATE
    _STATE = build_index(data_root, split, hash_features, df_cap_fraction, matrix_chunk)
    indexed_seconds = time.monotonic() - started
    source1_ids, _, _, _, index_t = _STATE
    print(json.dumps({"stage": "reverse_index", "split": split,
                      "source1": len(source1_ids), "nnz": int(index_t.nnz),
                      "seconds": round(indexed_seconds, 2)}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    processed = 0
    pairs = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for source in (2, 3):
            records = iter_records(source_path(data_root, split, source), source,
                                   max_rows=limit_targets)
            batches = _batches(records, batch_size)
            if workers > 1 and os.name == "posix":
                with multiprocessing.get_context("fork").Pool(workers) as pool:
                    for count, result in pool.imap(_top1_batch, batches, chunksize=1):
                        writer.writerows(result)
                        pairs += len(result)
                        processed += count
                        if processed // (batch_size * 500) != (processed - count) // (batch_size * 500):
                            print(json.dumps({"stage": "reverse_progress", "source": source,
                                              "processed_approx": processed,
                                              "pairs": pairs,
                                              "seconds": round(time.monotonic() - started, 2)}), flush=True)
            else:
                for batch in batches:
                    count, result = _top1_batch(batch)
                    writer.writerows(result)
                    pairs += len(result)
                    processed += count
        handle.flush()
    os.replace(temporary, out)
    report = {"split": split, "source1_rows": len(source1_ids),
              "reverse_pairs": pairs, "index_seconds": round(indexed_seconds, 2),
              "total_seconds": round(time.monotonic() - started, 2),
              "hash_features": hash_features, "df_cap_fraction": df_cap_fraction,
              "batch_size": batch_size, "workers": workers,
              "limit_targets": limit_targets,
              "config_hash": config_hash,
              "source1_sha256": sha256_file(source_path(data_root, split, 1))}
    metadata = out.with_suffix(out.suffix + ".meta.json")
    metadata.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "reverse_complete", **report}), flush=True)
    _STATE = None
    return report


def run_blank_name(data_root: Path, split: str, out: Path, *, top_k: int = 20, workers: int = 4,
                   batch_size: int = 256, hash_features: int = 1 << 22,
                   df_cap_fraction: float = 0.01, matrix_chunk: int = 200_000,
                   limit_targets: int | None = None, config_hash: str | None = None) -> dict:
    """Top-k Source 1 names for every target whose normalized address is blank."""
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if top_k < 1 or workers < 1 or batch_size < 1 or matrix_chunk < 1 or hash_features < 2:
        raise ValueError("top-k, worker, batch, matrix, and feature counts must be positive")
    if not 0 < df_cap_fraction <= 1:
        raise ValueError("df_cap_fraction must be in (0, 1]")
    started = time.monotonic()
    global _STATE
    _STATE = (*build_index(data_root, split, hash_features, df_cap_fraction, matrix_chunk,
                           text=_name_text), top_k)
    indexed_seconds = time.monotonic() - started
    queried = pairs = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for source in (2, 3):
            records = (record for record in iter_records(source_path(data_root, split, source), source,
                                                         max_rows=limit_targets)
                       if not normalize_address(record.business_address))
            batches = islice_batches(records, batch_size, _name_text)
            pool = (multiprocessing.get_context("fork").Pool(workers)
                    if workers > 1 and os.name == "posix" else None)
            try:
                results = pool.imap(_topk_batch, batches, chunksize=1) if pool else map(_topk_batch, batches)
                for count, result in results:
                    writer.writerows(result)
                    queried += count
                    pairs += len(result)
            finally:
                if pool:
                    pool.close()
                    pool.join()
    os.replace(temporary, out)
    report = {"split": split, "channel": "reverse_blank_name", "blank_address_targets": queried,
              "reverse_pairs": pairs, "top_k": top_k, "index_seconds": round(indexed_seconds, 2),
              "total_seconds": round(time.monotonic() - started, 2),
              "hash_features": hash_features, "df_cap_fraction": df_cap_fraction,
              "batch_size": batch_size, "workers": workers, "limit_targets": limit_targets,
              "config_hash": config_hash,
              "source1_sha256": sha256_file(source_path(data_root, split, 1))}
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "reverse_blank_complete", **report}), flush=True)
    _STATE = None
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hash-features", type=int, default=1 << 22)
    parser.add_argument("--df-cap-fraction", type=float, default=0.002)
    parser.add_argument("--matrix-chunk", type=int, default=200_000)
    parser.add_argument("--limit-targets", type=int,
                        help="Smoke benchmark only; output is incomplete")
    parser.add_argument("--config-hash", help="SHA-256 of the canonical pipeline configuration")
    parser.add_argument("--blank-name-top-k", type=int,
                        help="Run the blank-address name-only pass with this top-k instead of top-1")
    args = parser.parse_args()
    if args.blank_name_top_k:
        run_blank_name(args.data_root, args.split, args.out, top_k=args.blank_name_top_k,
                       workers=args.workers, batch_size=args.batch_size,
                       hash_features=args.hash_features, df_cap_fraction=args.df_cap_fraction,
                       matrix_chunk=args.matrix_chunk, limit_targets=args.limit_targets,
                       config_hash=args.config_hash)
        return
    run(args.data_root, args.split, args.out, workers=args.workers,
        batch_size=args.batch_size, hash_features=args.hash_features,
        df_cap_fraction=args.df_cap_fraction, matrix_chunk=args.matrix_chunk,
        limit_targets=args.limit_targets, config_hash=args.config_hash)


if __name__ == "__main__":
    main()
