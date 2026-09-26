"""Compare fixed final-selection rules against saved per-source candidate stages.

This is a development diagnostic. Labels are read only after retrieval, and the
reported oracle F0.5 assumes a perfect downstream matcher among kept pairs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .candidates import _choose_rows, _final_rank, _load_sorted_source1, _rank, _read_part, _select_source1
    from .data import iter_truth, source_path
except ImportError:
    from candidates import _choose_rows, _final_rank, _load_sorted_source1, _rank, _read_part, _select_source1
    from data import iter_truth, source_path


def _score(row: tuple[str, str, str, str, str], rule: str) -> float:
    name, address = float(row[2]), float(row[3])
    if rule in {"current", "balanced"}:
        return _final_rank(name, address, row[4], rule)
    if rule == "cosine":
        return 0.55 * name + 0.45 * address
    if rule == "both_fields":
        return 0.45 * name + 0.35 * address + 0.2 * min(name, address)
    raise ValueError(f"unknown score rule: {rule}")


def _choose(rows2: list, rows3: list, cap: int, quota: int, rule: str) -> list:
    if rule in {"current", "balanced"} and quota == cap // 2:
        return _choose_rows(rows2, rows3, cap, rule)

    def ordered(rows: list) -> list:
        return sorted(rows, key=lambda row: (-_score(row, rule), row[1]))

    left, right = ordered(rows2), ordered(rows3)
    kept = left[:quota] + right[:quota]
    remainder = left[quota:] + right[quota:]
    remainder.sort(key=lambda row: (-_score(row, rule), row[1]))
    return kept + remainder[:cap - len(kept)]


def evaluate(work_dir: Path, data_root: Path, sample_split: str,
             cap: int = 64, quotas: tuple[int, ...] = (0, 16, 24, 32),
             rules: tuple[str, ...] = ("current", "cosine", "balanced", "both_fields"),
             stage_limit: int | None = None) -> dict:
    if cap < 1 or any(quota < 0 or quota > cap // 2 for quota in quotas):
        raise ValueError("cap must be positive and quotas must fit both sources")
    if stage_limit is not None and stage_limit < 1:
        raise ValueError("stage_limit must be positive")
    chosen = _select_source1(_load_sorted_source1(source_path(data_root, "train", 1), None), sample_split)
    selected = {record.entity_id for record in chosen}
    truth = {entity_id: set(matches) for entity_id, matches in
             iter_truth(data_root / "train" / "train_ground_truth.tsv") if entity_id in selected}
    if set(truth) != selected:
        raise ValueError("selected Source 1 IDs do not match ground truth")

    parts = []
    for source in (2, 3):
        files = sorted((work_dir / f"source{source}").glob("part_*.tsv.gz"))
        if not files or [path.name for path in files] != [f"part_{i:06d}.tsv.gz" for i in range(len(files))]:
            raise ValueError(f"source{source}: missing or misnamed stage chunks")
        parts.append(files)
    if len(parts[0]) != len(parts[1]):
        raise ValueError("Source 2 and Source 3 have different chunk counts")

    if not rules or any(rule not in {"current", "cosine", "balanced", "both_fields"} for rule in rules):
        raise ValueError("provide at least one known score rule")
    variants = [(rule, quota) for rule in rules for quota in quotas]
    totals = {variant: defaultdict(float) for variant in variants}
    seen: set[str] = set()
    for part2, part3 in zip(*parts, strict=True):
        groups2, groups3 = _read_part(part2), _read_part(part3)
        ids = set(groups2) | set(groups3)
        if not ids <= selected or ids & seen:
            raise ValueError(f"unexpected or repeated Source 1 ID in {part2} or {part3}")
        seen.update(ids)
        for entity_id in ids:
            matches = truth[entity_id]
            rows2, rows3 = groups2.get(entity_id, []), groups3.get(entity_id, [])
            if len({row[1] for row in rows2 + rows3}) != len(rows2) + len(rows3):
                raise ValueError(f"duplicate staged target for {entity_id}")
            if stage_limit is not None:
                def trim(rows):
                    if len(rows) <= stage_limit:
                        return rows
                    return sorted(rows, key=lambda row: (
                        -_rank(float(row[2]), float(row[3]), row[4]), row[1]))[:stage_limit]
                rows2, rows3 = trim(rows2), trim(rows3)
            for variant in variants:
                rule, quota = variant
                kept = _choose(rows2, rows3, cap, quota, rule)
                hits = sum(row[1] in matches for row in kept)
                item = totals[variant]
                item["candidate_pairs"] += len(kept)
                item["covered_true_links"] += hits
                item["complete_set_rows"] += hits == len(matches)
                if not matches:
                    item["oracle_f0_5_sum"] += 1.0
                elif hits:
                    recall = hits / len(matches)
                    item["oracle_f0_5_sum"] += 1.25 * recall / (0.25 + recall)
    for entity_id in selected - seen:
        if not truth[entity_id]:
            for variant in variants:
                totals[variant]["complete_set_rows"] += 1
                totals[variant]["oracle_f0_5_sum"] += 1.0

    links = sum(map(len, truth.values()))
    results = []
    for (rule, quota), item in totals.items():
        hits = int(item["covered_true_links"])
        results.append({"score_rule": rule, "source_quota": quota,
                        "covered_true_links": hits,
                        "true_link_recall": hits / links if links else 1.0,
                        "candidate_pairs": int(item["candidate_pairs"]),
                        "complete_set_coverage_all": item["complete_set_rows"] / len(selected),
                        "oracle_macro_f0_5": item["oracle_f0_5_sum"] / len(selected)})
    results.sort(key=lambda row: (-row["oracle_macro_f0_5"], -row["true_link_recall"],
                                  row["candidate_pairs"], row["score_rule"], row["source_quota"]))
    return {"source1_rows": len(selected), "true_links": links, "cap": cap,
            "stage_limit_per_source": stage_limit,
            "stage_chunks_per_source": len(parts[0]), "variants": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--sample-split", required=True, choices=("dev", "holdout"))
    parser.add_argument("--cap", type=int, default=64)
    parser.add_argument("--quotas", type=int, nargs="+", default=[0, 16, 24, 32])
    parser.add_argument("--rules", nargs="+", choices=("current", "cosine", "balanced", "both_fields"),
                        default=["current", "cosine", "balanced", "both_fields"])
    parser.add_argument("--stage-limit", type=int,
                        help="Recreate a shallower first ranking cut from a deeper saved stage")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(args.work_dir, args.data_root, args.sample_split, args.cap,
                      tuple(args.quotas), tuple(args.rules), args.stage_limit)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
