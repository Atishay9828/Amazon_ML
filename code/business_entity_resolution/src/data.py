"""Strict TSV loading and reproducible dataset auditing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


@dataclass(frozen=True, slots=True)
class Record:
    entity_id: str
    business_name: str
    business_address: str
    country: str


def source_path(data_root: Path, split: str, source: int) -> Path:
    if split not in {"train", "test"} or source not in {1, 2, 3}:
        raise ValueError("split must be train/test and source must be 1/2/3")
    return data_root / split / f"{split}_source{source}.tsv"


def iter_records(path: Path, source: int, max_rows: int | None = None) -> Iterator[Record]:
    """Yield records while checking every visited row and ID."""
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"{path}: empty file") from exc
        if header != SOURCE_COLUMNS:
            raise ValueError(f"{path}: expected header {SOURCE_COLUMNS}, found {header}")
        for line, row in enumerate(reader, start=2):
            if len(row) != 4:
                raise ValueError(f"{path}:{line}: expected 4 tab-separated columns, got {len(row)}")
            entity_id = row[0]
            if not entity_id.startswith(f"S{source}-") or len(entity_id) <= 3:
                raise ValueError(f"{path}:{line}: invalid source-{source} ID {entity_id!r}")
            if entity_id in seen:
                raise ValueError(f"{path}:{line}: duplicate ID {entity_id!r}")
            seen.add(entity_id)
            yield Record(*row)
            if max_rows is not None and len(seen) >= max_rows:
                return


def iter_truth(path: Path) -> Iterator[tuple[str, tuple[str, ...]]]:
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"{path}: empty file") from exc
        if header != TRUTH_COLUMNS:
            raise ValueError(f"{path}: expected header {TRUTH_COLUMNS}, found {header}")
        for line, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"{path}:{line}: expected 2 tab-separated columns, got {len(row)}")
            source1_id, ids = row
            if not source1_id.startswith("S1-") or source1_id in seen:
                raise ValueError(f"{path}:{line}: invalid or duplicate Source 1 ID {source1_id!r}")
            seen.add(source1_id)
            matched = tuple(ids.split(",")) if ids else ()
            if len(matched) != len(set(matched)) or any(not mid.startswith(("S2-", "S3-")) for mid in matched):
                raise ValueError(f"{path}:{line}: invalid or repeated target ID")
            yield source1_id, matched


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(data_root: Path, include_hashes: bool = True) -> dict:
    """Audit source/label integrity. Uses ID-country maps, not record texts."""
    result: dict = {"files": {}, "truth": {}}
    train_source1_country: dict[str, str] = {}
    train_target_country: dict[str, str] = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            path = source_path(data_root, split, source)
            countries: Counter[str] = Counter()
            missing: Counter[str] = Counter()
            lengths = {"business_name": 0, "business_address": 0}
            maximum = {"business_name": 0, "business_address": 0}
            count = 0
            for record in iter_records(path, source):
                count += 1
                countries[record.country] += 1
                for field in ("business_name", "business_address", "country"):
                    value = getattr(record, field)
                    if not value.strip():
                        missing[field] += 1
                    if field in lengths:
                        lengths[field] += len(value)
                        maximum[field] = max(maximum[field], len(value))
                if split == "train":
                    if source == 1:
                        train_source1_country[record.entity_id] = record.country
                    else:
                        train_target_country[record.entity_id] = record.country
            result["files"][str(path.relative_to(data_root))] = {
                "rows": count,
                "countries": dict(sorted(countries.items())),
                "blank_fields": dict(sorted(missing.items())),
                "mean_lengths": {k: round(v / count, 3) if count else 0 for k, v in lengths.items()},
                "max_lengths": maximum,
                "bytes": path.stat().st_size,
                **({"sha256": sha256_file(path)} if include_hashes else {}),
            }
    truth_path = data_root / "train" / "train_ground_truth.tsv"
    truth_ids: set[str] = set()
    cardinality: Counter[int] = Counter()
    invalid_source1: list[str] = []
    invalid_targets: list[str] = []
    cross_country = 0
    total_links = 0
    for source1_id, matches in iter_truth(truth_path):
        truth_ids.add(source1_id)
        cardinality[len(matches)] += 1
        country = train_source1_country.get(source1_id)
        if country is None and len(invalid_source1) < 10:
            invalid_source1.append(source1_id)
        for target_id in matches:
            total_links += 1
            target_country = train_target_country.get(target_id)
            if target_country is None:
                if len(invalid_targets) < 10:
                    invalid_targets.append(target_id)
            elif country is not None and country != target_country:
                cross_country += 1
    result["truth"] = {
        "rows": len(truth_ids),
        "links": total_links,
        "cardinality": {str(k): v for k, v in sorted(cardinality.items())},
        "max_matches": max(cardinality, default=0),
        "missing_source1_label_count": len(train_source1_country.keys() - truth_ids),
        "invalid_source1_examples": invalid_source1,
        "invalid_target_examples": invalid_targets,
        "cross_country_links": cross_country,
        "sha256": sha256_file(truth_path) if include_hashes else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the supplied challenge TSVs")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--no-hashes", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.data_root, not args.no_hashes), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
