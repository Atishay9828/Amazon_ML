"""Compare true-link coverage at several per-source staged ranking cutoffs."""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
from pathlib import Path

try:
    from .candidates import HEADER, _load_sorted_source1, _rank, _select_source1
    from .data import iter_truth, source_path
except ImportError:
    from candidates import HEADER, _load_sorted_source1, _rank, _select_source1
    from data import iter_truth, source_path


def curve(work_dir: Path, data_root: Path, sample_split: str,
          cutoffs: list[int]) -> dict:
    if not cutoffs or any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("provide positive stage cutoffs")
    cutoffs = sorted(set(cutoffs))
    chosen = _select_source1(_load_sorted_source1(source_path(data_root, "train", 1), None), sample_split)
    ids = {record.entity_id for record in chosen}
    truth = {entity_id: set(matches) for entity_id, matches in
             iter_truth(data_root / "train" / "train_ground_truth.tsv") if entity_id in ids}
    if len(truth) != len(ids):
        raise ValueError("ground truth does not cover selected Source 1 records")
    link_counts = {cutoff: 0 for cutoff in cutoffs}
    pair_counts = {cutoff: 0 for cutoff in cutoffs}
    largest_group = 0
    for source in (2, 3):
        parts = sorted((work_dir / f"source{source}").glob("part_*.tsv.gz"))
        if not parts or [part.name for part in parts] != [f"part_{i:06d}.tsv.gz" for i in range(len(parts))]:
            raise ValueError(f"source{source}: missing or misnamed stage chunks")
        seen_source1 = set()
        for part in parts:
            with gzip.open(part, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                if tuple(next(reader)) != HEADER:
                    raise ValueError(f"{part}: invalid header")
                for entity_id, group in itertools.groupby(reader, key=lambda row: row[0]):
                    if entity_id not in truth or entity_id in seen_source1:
                        raise ValueError(f"{part}: invalid or repeated Source 1 entity {entity_id}")
                    seen_source1.add(entity_id)
                    rows = list(group)
                    target_ids = [row[1] for row in rows]
                    if (any(len(row) != 5 or not row[1].startswith(f"S{source}-") for row in rows)
                            or len(set(target_ids)) != len(rows)):
                        raise ValueError(f"{part}: invalid or duplicate target")
                    largest_group = max(largest_group, len(rows))
                    ranked = sorted(rows, key=lambda row: (-_rank(float(row[2]), float(row[3]), row[4]), row[1]))
                    for cutoff in cutoffs:
                        prefix = ranked[:cutoff]
                        pair_counts[cutoff] += len(prefix)
                        link_counts[cutoff] += sum(row[1] in truth[entity_id] for row in prefix)
    links = sum(map(len, truth.values()))
    return {"source1_rows": len(ids), "true_links": links, "largest_staged_source_group": largest_group,
            "cutoffs_per_source": [
                {"cutoff": cutoff, "pairs": pair_counts[cutoff], "covered_true_links": link_counts[cutoff],
                 "true_link_recall": link_counts[cutoff] / links if links else 1.0}
                for cutoff in cutoffs]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--sample-split", required=True, choices=("dev", "holdout"))
    parser.add_argument("--cutoffs", required=True, type=int, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = curve(args.work_dir, args.data_root, args.sample_split, args.cutoffs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
