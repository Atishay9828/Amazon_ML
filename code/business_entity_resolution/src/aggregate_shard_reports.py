"""Combine disjoint full-training shard reports with exact macro denominators."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

try:
    from .data import iter_records, iter_truth, source_path
except ImportError:
    from data import iter_records, iter_truth, source_path


def aggregate(data_root: Path, report_paths: list[Path], target_count: int,
              cap: int, log_paths: list[Path] | None = None) -> dict:
    if not report_paths or target_count < 1 or cap < 1:
        raise ValueError("reports, target_count, and cap must be positive")
    shard_count = len(report_paths)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    source1_ids = sorted(record.entity_id for record in iter_records(source_path(data_root, "train", 1), 1))
    boundaries = [source1_ids[len(source1_ids) * index // shard_count]
                  for index in range(1, shard_count)]
    truth_rows = [0] * shard_count
    nonempty_rows = [0] * shard_count
    true_links = [0] * shard_count
    for entity_id, matches in iter_truth(data_root / "train" / "train_ground_truth.tsv"):
        shard = bisect.bisect_right(boundaries, entity_id)
        truth_rows[shard] += 1
        nonempty_rows[shard] += bool(matches)
        true_links[shard] += len(matches)
    if sum(truth_rows) != len(source1_ids):
        raise ValueError("ground truth does not cover every Source 1 record")
    for index, report in enumerate(reports):
        expected_rows = len(source1_ids) * (index + 1) // shard_count - len(source1_ids) * index // shard_count
        if (report["source1_rows"] != expected_rows or truth_rows[index] != expected_rows or
                report["true_links"] != true_links[index]):
            raise ValueError(f"report {index} does not match its sorted Source 1 shard")

    rows = sum(truth_rows)
    links = sum(true_links)
    pairs = sum(report["candidate_pairs"] for report in reports)
    covered = sum(round(report["true_link_recall"] * true_links[index])
                  for index, report in enumerate(reports))
    complete_all = sum(round(report["complete_set_coverage_all"] * truth_rows[index])
                       for index, report in enumerate(reports))
    complete_nonempty = sum(round(report["complete_set_coverage_nonempty"] * nonempty_rows[index])
                            for index, report in enumerate(reports))
    nonempty = sum(nonempty_rows)
    oracle_sum = sum(report["oracle_macro_f0_5"] * truth_rows[index]
                     for index, report in enumerate(reports))
    result = {
        "source1_rows": rows,
        "target_rows": target_count,
        "true_links": links,
        "covered_true_links": covered,
        "candidate_pairs": pairs,
        "true_link_recall": covered / links if links else 1.0,
        "complete_set_coverage_all": complete_all / rows if rows else 1.0,
        "complete_set_coverage_nonempty": complete_nonempty / nonempty if nonempty else 1.0,
        "oracle_macro_f0_5": oracle_sum / rows if rows else 1.0,
        "candidate_mean": pairs / rows if rows else 0.0,
        "candidate_p95": cap if all(report["candidate_p95"] == cap for report in reports) else None,
        "zero_candidate_rows": sum(report["zero_candidate_rows"] for report in reports),
        "singleton_rows": rows - nonempty,
        "pair_reduction_ratio": 1 - pairs / (rows * target_count),
        "shard_count": shard_count,
    }
    if log_paths:
        if len(log_paths) != shard_count:
            raise ValueError("provide one Kaggle log per report")
        elapsed = []
        peaks = []
        for path in log_paths:
            entries = json.loads(path.read_text(encoding="utf-8"))
            completed = []
            for entry in entries:
                try:
                    message = json.loads(entry.get("data", ""))
                except json.JSONDecodeError:
                    continue
                if "peak_rss_gb" in message:
                    peaks.append(float(message["peak_rss_gb"]))
                if message.get("stage") == "complete":
                    completed.append(float(entry["time"]))
            if len(completed) != 1:
                raise ValueError(f"expected one complete event in {path}")
            elapsed.append(completed[0])
        result["shard_elapsed_seconds"] = elapsed
        result["max_shard_elapsed_seconds"] = max(elapsed)
        result["max_reported_process_peak_rss_gb"] = max(peaks) if peaks else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True,
                        help="One report per sorted shard, in shard-index order")
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--logs", type=Path, nargs="+", help="Optional Kaggle logs in shard-index order")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.data_root, args.reports, args.target_count, args.cap, args.logs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
