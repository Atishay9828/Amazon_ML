"""Merge contiguous sorted Source 1 candidate shards with streaming checks."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from candidates import HEADER


def merge(inputs: list[Path], out: Path) -> int:
    if not inputs:
        raise ValueError("at least one shard is required")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    previous = ("", "")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for path in inputs:
            with path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.reader(source, delimiter="\t")
                if tuple(next(reader, ())) != HEADER:
                    raise ValueError(f"invalid candidate header in {path}")
                for row in reader:
                    if len(row) != len(HEADER):
                        raise ValueError(f"invalid row width in {path}")
                    pair = row[0], row[1]
                    if pair <= previous:
                        raise ValueError(f"shard order or duplicate pair at {pair} in {path}")
                    writer.writerow(row)
                    previous = pair
                    count += 1
    os.replace(temporary, out)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path, help="shards in ascending shard-index order")
    args = parser.parse_args()
    print(f"merged {merge(args.inputs, args.out):,} candidate pairs into {args.out}")


if __name__ == "__main__":
    main()
