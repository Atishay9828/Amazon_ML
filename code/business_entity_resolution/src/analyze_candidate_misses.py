"""Summarize missed development links without exporting challenge records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from candidates import HEADER, _load_sorted_source1, _select_source1
from data import iter_records, iter_truth, source_path
from normalization import normalize_address, normalize_name


def overlap(left: str, right: str) -> bool:
    return bool(set(left.split()) & set(right.split()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--sample-split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    chosen = _select_source1(_load_sorted_source1(source_path(args.data_root, "train", 1), None), args.sample_split)
    source1 = {record.entity_id: record for record in chosen}
    truth = {entity_id: set(matches) for entity_id, matches in iter_truth(args.data_root / "train" / "train_ground_truth.tsv")
             if entity_id in source1}
    covered: dict[str, set[str]] = {entity_id: set() for entity_id in source1}
    counts: Counter[tuple[str, str]] = Counter()
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader)) != HEADER:
            raise ValueError("candidate header mismatch")
        for row in reader:
            if len(row) != 5 or row[0] not in covered:
                raise ValueError("unexpected candidate row")
            counts[(row[0], row[1][:2])] += 1
            if row[1] in truth[row[0]]:
                covered[row[0]].add(row[1])
    misses = {(entity_id, target_id) for entity_id, ids in truth.items() for target_id in ids - covered[entity_id]}
    needed = {target for _, target in misses}
    targets = {}
    for source in (2, 3):
        for record in iter_records(source_path(args.data_root, "train", source), source):
            if record.entity_id in needed:
                targets[record.entity_id] = record
    if len(targets) != len(needed):
        raise ValueError("some missed targets are absent from the source files")

    categories = Counter()
    by_source = Counter()
    by_country = Counter()
    source_candidate_counts = Counter()
    examples = []
    for entity_id, target_id in sorted(misses):
        left, right = source1[entity_id], targets[target_id]
        ln, rn = normalize_name(left.business_name), normalize_name(right.business_name)
        la, ra = normalize_address(left.business_address), normalize_address(right.business_address)
        name_shared = overlap(ln, rn)
        address_shared = overlap(la, ra)
        category = ("name_token" if name_shared else "no_name_token") + "+" + ("address_token" if address_shared else "no_address_token")
        categories[category] += 1
        by_source[target_id[:2]] += 1
        by_country[left.country] += 1
        source_candidate_counts[counts[(entity_id, target_id[:2])]] += 1
        if len(examples) < 32:
            examples.append({"source1_id": entity_id, "target_id": target_id,
                             "source1_name": left.business_name, "target_name": right.business_name,
                             "source1_address": left.business_address, "target_address": right.business_address,
                             "category": category})
    report = {"sample_split": args.sample_split, "missed_links": len(misses),
              "categories": dict(sorted(categories.items())), "by_source": dict(sorted(by_source.items())),
              "by_country": dict(sorted(by_country.items())),
              "misses_at_source_quota_64": source_candidate_counts[64],
              "misses_below_source_quota_64": sum(value for count, value in source_candidate_counts.items() if count < 64),
              "source_candidate_count_distribution_for_misses": dict(sorted(source_candidate_counts.items())),
              "examples": examples}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
