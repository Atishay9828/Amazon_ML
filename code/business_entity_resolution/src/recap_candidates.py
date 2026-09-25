"""Reduce a cap-64 long-form candidate table to a tested smaller cap."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from candidates import HEADER, _choose_rows


def recap(source_path: Path, out: Path, cap: int) -> int:
    if cap < 1:
        raise ValueError("cap must be positive")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    count = 0
    with source_path.open("r", encoding="utf-8", newline="") as source, \
            temporary.open("w", encoding="utf-8", newline="") as target:
        reader = csv.reader(source, delimiter="\t")
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        if tuple(next(reader, ())) != HEADER:
            raise ValueError("invalid candidate header")
        writer.writerow(HEADER)
        current = ""
        left: list[list[str]] = []
        right: list[list[str]] = []
        previous = ("", "")

        def flush() -> int:
            chosen = _choose_rows(left, right, cap)
            writer.writerows(chosen)
            return len(chosen)

        for row in reader:
            if len(row) != len(HEADER):
                raise ValueError("invalid row width")
            pair = row[0], row[1]
            if pair <= previous:
                raise ValueError("candidate pairs must be unique and sorted")
            previous = pair
            if current and row[0] != current:
                count += flush()
                left, right = [], []
            current = row[0]
            if row[1].startswith("S2-"):
                left.append(row)
            elif row[1].startswith("S3-"):
                right.append(row)
            else:
                raise ValueError("candidate target is not from Source 2 or 3")
        if current:
            count += flush()
    os.replace(temporary, out)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cap", required=True, type=int)
    args = parser.parse_args()
    print(f"wrote {recap(args.source, args.out, args.cap):,} pairs to {args.out}")


if __name__ == "__main__":
    main()
