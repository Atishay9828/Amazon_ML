"""Score candidate recall and oracle macro F0.5 for a fixed train selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from candidates import _load_sorted_source1, _select_source1, evaluate_retrieval
from data import source_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--sample-split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source1 = _load_sorted_source1(source_path(args.data_root, "train", 1), None)
    chosen = _select_source1(source1, args.sample_split)
    selected_ids = {record.entity_id for record in chosen}
    report = evaluate_retrieval(args.input, args.data_root / "train" / "train_ground_truth.tsv", selected_ids)
    report["sample_split"] = args.sample_split
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
