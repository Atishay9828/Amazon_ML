"""Reassemble a new candidate cap from downloaded Kaggle retrieval chunks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from candidates import (HEADER, RETRIEVAL_VERSION, _choose_rows,
                        _load_sorted_source1, _part_path, _read_part,
                        _select_source1)
from data import source_path


def assemble(data_root: Path, split: str, work_dir: Path, out: Path, cap: int,
             sample_split: str | None = None, shard_index: int = 0,
             shard_count: int = 1) -> int:
    if not 1 <= cap <= 64:
        raise ValueError("staged retrieval supports caps 1 through 64")
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = {"retrieval_version": RETRIEVAL_VERSION, "split": split,
                "sample_split": sample_split, "shard_index": shard_index,
                "shard_count": shard_count, "limit_source1": None,
                "limit_targets": None}
    for key, value in expected.items():
        default = 0 if key == "shard_index" else 1 if key == "shard_count" else None
        if manifest.get(key, default) != value:
            raise ValueError(f"staged manifest mismatch for {key}: {manifest.get(key, default)!r} != {value!r}")
    chunk_size = manifest["settings"]["query_chunk"]
    source1 = _select_source1(_load_sorted_source1(source_path(data_root, split, 1), None),
                              sample_split, shard_index, shard_count)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for chunk_number, start in enumerate(range(0, len(source1), chunk_size)):
            groups2 = _read_part(_part_path(work_dir, 2, chunk_number))
            groups3 = _read_part(_part_path(work_dir, 3, chunk_number))
            for record in source1[start:start + chunk_size]:
                chosen = _choose_rows(groups2.get(record.entity_id, []),
                                      groups3.get(record.entity_id, []), cap)
                writer.writerows(chosen)
                count += len(chosen)
    os.replace(temporary, out)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--sample-split", choices=("dev", "holdout"))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    print(f"assembled {assemble(args.data_root, args.split, args.work_dir, args.out, args.cap, args.sample_split, args.shard_index, args.shard_count):,} pairs")


if __name__ == "__main__":
    main()
