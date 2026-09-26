"""Convert a sorted long-form candidate table into the official wide candidate_pairs.tsv.

Every Source 1 record of the split gets exactly one row, with an empty list when
retrieval found no candidate. The conversion streams the sorted long-form file, so
memory stays small for tables with more than 100 million pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

try:
    from .data import iter_records, source_path
except ImportError:
    from data import iter_records, source_path

WIDE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def export(long_path: Path, data_root: Path, split: str, out: Path) -> dict:
    remaining = {record.entity_id for record in iter_records(source_path(data_root, split, 1), 1)}
    source1_rows = len(remaining)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    pairs = 0
    with long_path.open("r", encoding="utf-8", newline="") as source, \
            temporary.open("w", encoding="utf-8", newline="") as target:
        reader = csv.reader(source, delimiter="\t")
        header = next(reader, ())
        if tuple(header[:2]) != ("source1_entity_id", "candidate_entity_id"):
            raise ValueError(f"{long_path}: not a long-form candidate table")
        target.write("\t".join(WIDE_HEADER) + "\n")
        current, ids, previous = "", [], ("", "")

        def flush() -> None:
            if current not in remaining:
                raise ValueError(f"{long_path}: Source 1 ID {current} is absent from {split} or repeated")
            remaining.discard(current)
            target.write(f"{current}\t{','.join(ids)}\n")

        for row in reader:
            pair = (row[0], row[1])
            if pair <= previous:
                raise ValueError(f"{long_path}: pairs are not unique and sorted at {pair}")
            if not row[1].startswith(("S2-", "S3-")):
                raise ValueError(f"{long_path}: candidate {row[1]} is not a Source 2/3 ID")
            previous = pair
            if row[0] != current:
                if current:
                    flush()
                current, ids = row[0], []
            ids.append(row[1])
            pairs += 1
        if current:
            flush()
        for entity_id in sorted(remaining):
            target.write(f"{entity_id}\t\n")
    os.replace(temporary, out)
    report = {"source1_rows": source1_rows, "candidate_pairs": pairs, "empty_rows": len(remaining),
              "long_form": str(long_path), "out": str(out)}
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", required=True, type=Path, help="Sorted long-form candidate TSV")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--out", required=True, type=Path, help="Wide candidate_pairs.tsv to write")
    args = parser.parse_args()
    export(args.long, args.data_root, args.split, args.out)


if __name__ == "__main__":
    main()
