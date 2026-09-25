"""Validate Person A's exact long-form handoff against the official sources."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from candidates import CHANNEL_ORDER, HEADER, _load_sorted_source1
from data import iter_records, source_path


def validate(path: Path, data_root: Path, split: str, cap: int) -> dict:
    if cap < 1:
        raise ValueError("cap must be positive")
    records = iter(_load_sorted_source1(source_path(data_root, split, 1), None))
    current_record = next(records, None)
    unseen_targets: set[str] = set()
    pair_count = 0
    represented = 0
    previous_pair = ("", "")
    current_id = ""
    per_entity = 0
    allowed_channels = set(CHANNEL_ORDER)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader, ())) != HEADER:
            raise ValueError("invalid five-column header")
        for row in reader:
            if len(row) != 5:
                raise ValueError(f"invalid five-column row at pair {pair_count + 1}")
            source1_id, target_id, name_score, address_score, channels = row
            pair = source1_id, target_id
            if pair <= previous_pair:
                raise ValueError(f"duplicate or unsorted pair: {pair}")
            previous_pair = pair
            if source1_id != current_id:
                represented += 1
                per_entity = 0
                current_id = source1_id
                while current_record is not None and current_record.entity_id < source1_id:
                    current_record = next(records, None)
                if current_record is None or current_record.entity_id != source1_id:
                    raise ValueError(f"unknown Source 1 ID: {source1_id}")
            per_entity += 1
            if per_entity > cap:
                raise ValueError(f"Source 1 entity exceeds cap {cap}: {source1_id}")
            if not target_id.startswith(("S2-", "S3-")):
                raise ValueError(f"invalid target source: {target_id}")
            unseen_targets.add(target_id)
            for score in (name_score, address_score):
                value = float(score)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"invalid cosine score at {pair}")
            channel_set = channels.split("|")
            if not channel_set or len(channel_set) != len(set(channel_set)) or not set(channel_set) <= allowed_channels:
                raise ValueError(f"invalid retrieval channels at {pair}")
            pair_count += 1
    for source in (2, 3):
        for record in iter_records(source_path(data_root, split, source), source):
            unseen_targets.discard(record.entity_id)
    if unseen_targets:
        raise ValueError(f"{len(unseen_targets)} candidate target IDs are absent; example: {min(unseen_targets)}")
    return {"candidate_pairs": pair_count, "represented_source1": represented,
            "cap": cap, "target_ids_valid": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--cap", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.input, args.data_root, args.split, args.cap), indent=2))


if __name__ == "__main__":
    main()
