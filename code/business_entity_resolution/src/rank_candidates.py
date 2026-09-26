"""Train a candidate-shortlist ranker and apply it to staged pair chunks.

Training labels come only from the dedicated ranktrain Source 1 selection.
Inference reads no labels and preserves the five-column Person A contract.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from .candidates import (CHANNEL_ORDER, HEADER, _load_sorted_source1,
                             _memory_state, _part_path, _select_source1, _rank)
    from .data import Record, iter_records, iter_truth, source_path
    from .normalization import (compact_name, informative_address_tokens,
                                informative_name_tokens, name_token_signature,
                                normalize_address, normalize_name)
except ImportError:
    from candidates import (CHANNEL_ORDER, HEADER, _load_sorted_source1,
                            _memory_state, _part_path, _select_source1, _rank)
    from data import Record, iter_records, iter_truth, source_path
    from normalization import (compact_name, informative_address_tokens,
                               informative_name_tokens, name_token_signature,
                               normalize_address, normalize_name)


BASE_FEATURES = ("name_cosine", "address_cosine", "prod", "mx", "mn",
                 "is_s3", "base_rank", "base_pos", "name_pos", "addr_pos",
                 "source_count", "n_channels", "name_jac", "addr_jac",
                 "num_jac", "num_any", "num_conflict", "sig_eq",
                 "compact_eq", "t_addr_blank", "country_eq")
FEATURE_NAMES = BASE_FEATURES + tuple("ch_" + channel for channel in CHANNEL_ORDER)
_NUMBER = re.compile(r"(?<!\w)(\d+)(?:st|nd|rd|th)?\b")


@dataclass(frozen=True)
class Prepared:
    name_tokens: frozenset[str]
    address_tokens: frozenset[str]
    numbers: frozenset[str]
    signature: str
    compact: str
    address_blank: bool
    country: str


def _prepare(record: Record) -> Prepared:
    name = normalize_name(record.business_name)
    address = normalize_address(record.business_address)
    return Prepared(frozenset(informative_name_tokens(name)),
                    frozenset(informative_address_tokens(address)),
                    frozenset(_NUMBER.findall(address)),
                    name_token_signature(name), compact_name(name),
                    not bool(address), record.country)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


class FeatureContext:
    def __init__(self, records: dict[str, Record]):
        self.records = records

    @lru_cache(maxsize=300_000)
    def prepared(self, entity_id: str) -> Prepared:
        return _prepare(self.records[entity_id])


def _positions(rows: list[tuple[str, str, str, str, str]],
               values: list[float]) -> list[int]:
    ordered = sorted(range(len(rows)), key=lambda i: (-values[i], rows[i][1]))
    result = [0] * len(rows)
    for position, index in enumerate(ordered, start=1):
        result[index] = position
    return result


def features_for_group(source1_id: str,
                       rows2: list[tuple[str, str, str, str, str]],
                       rows3: list[tuple[str, str, str, str, str]],
                       context: FeatureContext) -> tuple[list[tuple[str, str, str, str, str]], np.ndarray]:
    source1 = context.prepared(source1_id)
    rows = rows2 + rows3
    matrix = np.empty((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    for source_rows, offset, is_s3 in ((rows2, 0, False), (rows3, len(rows2), True)):
        if not source_rows:
            continue
        names = [float(row[2]) for row in source_rows]
        addresses = [float(row[3]) for row in source_rows]
        bases = [_rank(n, a, row[4]) for row, n, a in
                 zip(source_rows, names, addresses, strict=True)]
        base_positions = _positions(source_rows, bases)
        name_positions = _positions(source_rows, names)
        address_positions = _positions(source_rows, addresses)
        for local, row in enumerate(source_rows):
            target = context.prepared(row[1])
            n, a = names[local], addresses[local]
            channels = set(row[4].split("|"))
            common_numbers = source1.numbers & target.numbers
            values = (n, a, n * a, max(n, a), min(n, a),
                      float(is_s3), bases[local], base_positions[local],
                      name_positions[local], address_positions[local],
                      len(source_rows), len(channels),
                      _jaccard(source1.name_tokens, target.name_tokens),
                      _jaccard(source1.address_tokens, target.address_tokens),
                      _jaccard(source1.numbers, target.numbers),
                      bool(common_numbers),
                      bool(source1.numbers and target.numbers and not common_numbers),
                      bool(source1.signature and source1.signature == target.signature),
                      bool(len(source1.compact) >= 8 and source1.compact == target.compact),
                      target.address_blank, source1.country == target.country)
            matrix[offset + local] = values + tuple(channel in channels for channel in CHANNEL_ORDER)
    return rows, matrix


def select_ranked(rows2: list[tuple[str, str, str, str, str]],
                  rows3: list[tuple[str, str, str, str, str]],
                  probabilities: np.ndarray, cap: int) -> list[tuple[str, str, str, str, str]]:
    if cap < 1 or len(probabilities) != len(rows2) + len(rows3):
        raise ValueError("invalid cap or probability count")
    rows = rows2 + rows3
    left = sorted(range(len(rows2)), key=lambda i: (-probabilities[i], rows[i][1]))
    right = sorted(range(len(rows2), len(rows)),
                   key=lambda i: (-probabilities[i], rows[i][1]))
    quota = cap // 2
    chosen = left[:quota] + right[:quota]
    remainder = left[quota:] + right[quota:]
    remainder.sort(key=lambda i: (-probabilities[i], rows[i][1]))
    chosen.extend(remainder[:cap - len(chosen)])
    return sorted((rows[index] for index in chosen), key=lambda row: row[1])


def _load_records(data_root: Path, split: str,
                  source1: list[Record]) -> dict[str, Record]:
    records = {record.entity_id: record for record in source1}
    for source in (2, 3):
        for record in iter_records(source_path(data_root, split, source), source):
            records[record.entity_id] = record
    return records


def _stage_manifest(work_dir: Path, split: str, sample_split: str | None,
                    shard_index: int, shard_count: int, min_stage_cap: int) -> dict:
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = {"split": split, "sample_split": sample_split,
                "shard_index": shard_index, "shard_count": shard_count,
                "limit_source1": None, "limit_targets": None}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"stage manifest {key} mismatch: {manifest.get(key)!r} != {value!r}")
    if manifest["settings"]["stage_cap"] < min_stage_cap:
        raise ValueError(f"stage cap below required {min_stage_cap}")
    if "config_hash" not in manifest:
        raise ValueError("staged manifest has no config hash field")
    return manifest


def _retrieval_signature(manifest: dict) -> dict:
    settings = manifest["settings"]
    signature = {key: value for key, value in settings.items()
                 if key not in {"extra_pairs_sha256", "query_chunk", "matrix_chunk"}}
    signature["has_reverse_top1"] = settings.get("extra_pairs_sha256") is not None
    return signature


def _read_group_part(path: Path) -> dict[str, list[tuple[str, str, str, str, str]]]:
    groups = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        if tuple(next(reader, ())) != HEADER:
            raise ValueError(f"{path}: invalid staged header")
        for row in reader:
            if len(row) != 5:
                raise ValueError(f"{path}: invalid staged row")
            groups.setdefault(row[0], []).append(tuple(row))
    return groups


def _iter_chunks(work_dir: Path, source1: list[Record], query_chunk: int):
    for chunk, start in enumerate(range(0, len(source1), query_chunk)):
        parts = []
        for source in (2, 3):
            path = _part_path(work_dir, source, chunk)
            if not path.is_file():
                raise FileNotFoundError(f"missing staged chunk: {path}")
            parts.append(_read_group_part(path))
        yield source1[start:start + query_chunk], parts[0], parts[1]


def train(data_root: Path, work_dir: Path, model_out: Path,
          max_negatives_per_source: int = 32,
          model_params: dict | None = None) -> dict:
    if max_negatives_per_source < 1:
        raise ValueError("max_negatives_per_source must be positive")
    started = time.monotonic()
    manifest = _stage_manifest(work_dir, "train", "ranktrain", 0, 1, 256)
    source1 = _select_source1(_load_sorted_source1(source_path(data_root, "train", 1), None),
                              "ranktrain")
    if len(source1) < 50_000:
        raise ValueError("ranker needs at least 50,000 disjoint training Source 1 records")
    selected = {record.entity_id for record in source1}
    truth = {entity_id: set(matches) for entity_id, matches in
             iter_truth(data_root / "train" / "train_ground_truth.tsv") if entity_id in selected}
    if len(truth) != len(source1):
        raise ValueError("ground truth does not cover ranker training sample")
    context = FeatureContext(_load_records(data_root, "train", source1))
    print(json.dumps({"stage": "ranker_records_loaded", "split": "train",
                      "records": len(context.records), **_memory_state()}), flush=True)
    matrices = []
    labels = []
    positives = 0
    for chunk, (batch, groups2, groups3) in enumerate(
            _iter_chunks(work_dir, source1, manifest["settings"]["query_chunk"]), start=1):
        for record in batch:
            left = groups2.get(record.entity_id, [])
            right = groups3.get(record.entity_id, [])
            if not left and not right:
                continue
            rows, matrix = features_for_group(record.entity_id, left, right, context)
            target_truth = truth[record.entity_id]
            keep = []
            for offset, source_rows in ((0, left), (len(left), right)):
                positions = list(range(offset, offset + len(source_rows)))
                positive = [i for i in positions if rows[i][1] in target_truth]
                negative = [i for i in positions if rows[i][1] not in target_truth]
                negative.sort(key=lambda i: (-matrix[i, BASE_FEATURES.index("base_rank")],
                                             rows[i][1]))
                keep.extend(positive + negative[:max_negatives_per_source])
            if keep:
                matrices.append(matrix[keep])
                group_labels = np.fromiter((rows[i][1] in target_truth for i in keep),
                                           dtype=np.uint8, count=len(keep))
                labels.append(group_labels)
                positives += int(group_labels.sum())
        if chunk % 5 == 0:
            print(json.dumps({"stage": "ranker_training_features", "chunks": chunk,
                              "seconds": round(time.monotonic() - started, 2),
                              **_memory_state()}), flush=True)
    if not matrices or positives == 0:
        raise ValueError("ranker training has no positive staged links")
    X = np.vstack(matrices)
    y = np.concatenate(labels)
    del matrices, labels, context
    model_params = model_params or {"max_iter": 300, "learning_rate": 0.1,
                                    "max_leaf_nodes": 63, "random_state": 0}
    model = HistGradientBoostingClassifier(**model_params)
    model.fit(X, y)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_out.with_suffix(model_out.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"model": model, "feature_names": FEATURE_NAMES,
                     "training_source1": len(source1),
                     "retrieval_signature": _retrieval_signature(manifest),
                     "config_hash": manifest["config_hash"],
                     "model_params": model_params,
                     "max_negatives_per_source": max_negatives_per_source}, handle)
    os.replace(temporary, model_out)
    report = {"training_source1": len(source1), "samples": int(len(y)),
              "positives": positives, "negatives": int(len(y) - positives),
              "features": list(FEATURE_NAMES),
              "max_negatives_per_source": max_negatives_per_source,
              "model_params": model_params,
              "seconds": round(time.monotonic() - started, 2), **_memory_state()}
    model_out.with_suffix(model_out.suffix + ".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "ranker_train_complete", **report}), flush=True)
    return report


def infer(data_root: Path, split: str, work_dir: Path, model_path: Path,
          out: Path, cap: int, sample_split: str | None = None,
          shard_index: int = 0, shard_count: int = 1) -> dict:
    if cap < 1:
        raise ValueError("cap must be positive")
    started = time.monotonic()
    manifest = _stage_manifest(work_dir, split, sample_split,
                               shard_index, shard_count, 256)
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if tuple(bundle["feature_names"]) != FEATURE_NAMES:
        raise ValueError("ranker feature schema differs from current code")
    if bundle["retrieval_signature"] != _retrieval_signature(manifest):
        raise ValueError("ranker was trained on different candidate-generation settings")
    if bundle.get("config_hash") != manifest.get("config_hash"):
        raise ValueError("ranker model and staged chunks have different config hashes")
    model = bundle["model"]
    source1 = _select_source1(_load_sorted_source1(source_path(data_root, split, 1), None),
                              sample_split, shard_index, shard_count)
    context = FeatureContext(_load_records(data_root, split, source1))
    print(json.dumps({"stage": "ranker_records_loaded", "split": split,
                      "records": len(context.records), **_memory_state()}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for chunk, (batch, groups2, groups3) in enumerate(
                _iter_chunks(work_dir, source1, manifest["settings"]["query_chunk"]), start=1):
            frames = []
            groups = []
            for record in batch:
                left = groups2.get(record.entity_id, [])
                right = groups3.get(record.entity_id, [])
                if not left and not right:
                    continue
                _, features = features_for_group(record.entity_id, left, right, context)
                frames.append(features)
                groups.append((left, right, len(features)))
            if not frames:
                continue
            probabilities = model.predict_proba(np.vstack(frames))[:, 1]
            offset = 0
            for left, right, length in groups:
                chosen = select_ranked(left, right, probabilities[offset:offset + length], cap)
                writer.writerows(chosen)
                count += len(chosen)
                offset += length
            if chunk % 20 == 0:
                print(json.dumps({"stage": "ranker_infer_progress", "chunks": chunk,
                                  "pairs": count,
                                  "seconds": round(time.monotonic() - started, 2),
                                  **_memory_state()}), flush=True)
    os.replace(temporary, out)
    report = {"split": split, "sample_split": sample_split, "source1_rows": len(source1),
              "candidate_pairs": count, "cap": cap,
              "seconds": round(time.monotonic() - started, 2),
              "model": str(model_path), **_memory_state()}
    print(json.dumps({"stage": "ranker_infer_complete", **report}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data-root", required=True, type=Path)
    train_parser.add_argument("--work-dir", required=True, type=Path)
    train_parser.add_argument("--model-out", required=True, type=Path)
    train_parser.add_argument("--max-negatives-per-source", type=int, default=32)
    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--data-root", required=True, type=Path)
    infer_parser.add_argument("--split", required=True, choices=("train", "test"))
    infer_parser.add_argument("--work-dir", required=True, type=Path)
    infer_parser.add_argument("--model", required=True, type=Path)
    infer_parser.add_argument("--out", required=True, type=Path)
    infer_parser.add_argument("--cap", type=int, default=32)
    infer_parser.add_argument("--sample-split", choices=("dev", "holdout", "ranktrain"))
    infer_parser.add_argument("--shard-index", type=int, default=0)
    infer_parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.data_root, args.work_dir, args.model_out,
              args.max_negatives_per_source)
    else:
        if args.sample_split and args.split != "train":
            parser.error("sample splits are only available for training data")
        infer(args.data_root, args.split, args.work_dir, args.model,
              args.out, args.cap, args.sample_split,
              args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
