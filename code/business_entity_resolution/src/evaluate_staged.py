"""Measure the proposal recall ceiling from downloaded per-source stage chunks."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

try:
    from .candidates import HEADER, _load_sorted_source1, _select_source1
    from .data import iter_truth, source_path
except ImportError:
    from candidates import HEADER, _load_sorted_source1, _select_source1
    from data import iter_truth, source_path


def evaluate(work_dir: Path, data_root: Path, sample_split: str) -> dict:
    chosen = _select_source1(_load_sorted_source1(source_path(data_root, "train", 1), None), sample_split)
    ids = {record.entity_id for record in chosen}
    truth = {entity_id: set(matches) for entity_id, matches in
             iter_truth(data_root / "train" / "train_ground_truth.tsv") if entity_id in ids}
    if len(truth) != len(ids):
        raise ValueError("ground truth does not cover every selected Source 1 entity")
    pairs = 0
    covered = 0
    seen = set()
    expected_parts = None
    for source in (2, 3):
        parts = sorted((work_dir / f"source{source}").glob("part_*.tsv.gz"))
        if not parts:
            raise ValueError(f"source{source}: no stage chunks found")
        numbers = [int(part.name.removeprefix("part_").removesuffix(".tsv.gz")) for part in parts]
        if numbers != list(range(len(parts))) or (expected_parts is not None and len(parts) != expected_parts):
            raise ValueError(f"source{source}: missing or inconsistent stage chunks")
        expected_parts = len(parts)
        for part in parts:
            with gzip.open(part, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                if tuple(next(reader)) != HEADER:
                    raise ValueError(f"{part}: invalid header")
                for row in reader:
                    if len(row) != 5 or row[0] not in truth or not row[1].startswith(f"S{source}-"):
                        raise ValueError(f"{part}: invalid candidate pair")
                    pair = (row[0], row[1])
                    if pair in seen:
                        raise ValueError(f"duplicate staged pair: {pair}")
                    seen.add(pair)
                    pairs += 1
                    covered += row[1] in truth[row[0]]
    links = sum(map(len, truth.values()))
    return {"source1_rows": len(ids), "true_links": links, "staged_pairs": pairs,
            "staged_true_links": covered,
            "staged_true_link_recall": covered / links if links else 1.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--sample-split", required=True, choices=("dev", "holdout"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = evaluate(args.work_dir, args.data_root, args.sample_split)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
