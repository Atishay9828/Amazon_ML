"""Bounded candidate retrieval for the Amazon ML entity-resolution challenge.

The long-form output is the exact pair set Person B must score. Intermediate
per-source/chunk files make the expensive retrieval resumable and permit cap
experiments without recomputing similarities. No external identity data is used.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

try:  # permit both `python -m src.candidates` and direct script execution
    from .data import Record, iter_records, iter_truth, source_path
    from .normalization import informative_address_tokens, informative_name_tokens, normalize_address, normalize_name
except ImportError:
    from data import Record, iter_records, iter_truth, source_path
    from normalization import informative_address_tokens, informative_name_tokens, normalize_address, normalize_name

HEADER = ("source1_entity_id", "candidate_entity_id", "name_cosine", "address_cosine", "retrieval_channels")
CHANNEL_ORDER = ("exact_name", "exact_address", "rare_name", "rare_address", "name_char", "address_char")
RETRIEVAL_VERSION = "rare-address-translit-vote-1"


@dataclass(frozen=True)
class Settings:
    name_k: int = 32
    address_k: int = 16
    cap: int = 32
    query_chunk: int = 2048
    matrix_chunk: int = 100_000
    hash_features: int = 1 << 20
    max_common_df: float = 0.05
    rare_max_df: int = 64
    rare_address_max_df: int = 128
    max_exact_block: int = 2_000
    indexed_grams: int = 8
    query_grams: int = 16
    max_probe_df: int = 2_000


@dataclass
class FieldIndex:
    vectorizer: HashingVectorizer
    tfidf: TfidfTransformer
    target: sparse.csr_matrix
    common: np.ndarray
    posting_offsets: np.ndarray
    posting_rows: np.ndarray

    def query(self, texts: list[str]) -> sparse.csr_matrix:
        matrix = self.vectorizer.transform(texts).tocsr()
        matrix.data[self.common[matrix.indices]] = 0
        matrix.eliminate_zeros()
        return self.tfidf.transform(matrix).tocsr()


def _field_index(texts: list[str], settings: Settings) -> FieldIndex:
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=(3, 5), n_features=settings.hash_features,
        alternate_sign=False, norm=None, dtype=np.float32,
    )
    blocks = [vectorizer.transform(texts[i:i + settings.matrix_chunk])
              for i in range(0, len(texts), settings.matrix_chunk)]
    matrix = sparse.vstack(blocks, format="csr", dtype=np.float32) if blocks else sparse.csr_matrix((0, settings.hash_features), dtype=np.float32)
    del blocks
    if matrix.shape[0] == 0:
        raise ValueError("target source contains no records")
    # Frequent character grams create enormous postings and little discrimination.
    # CSR has one entry per (record, hashed feature), so bincount gives document frequency.
    df = np.bincount(matrix.indices, minlength=settings.hash_features)
    common = df > max(2, int(settings.max_common_df * len(texts)))
    if common.any():
        matrix.data[common[matrix.indices]] = 0
        matrix.eliminate_zeros()
    tfidf = TfidfTransformer(norm="l2", use_idf=True, sublinear_tf=True)
    matrix = tfidf.fit_transform(matrix).tocsr().astype(np.float32)
    # Index only each target's rarest grams. A dense all-pairs sparse product
    # touches far too many common-character postings at challenge scale.
    capacity = len(texts) * settings.indexed_grams
    indexed_features = np.empty(capacity, dtype=np.int32)
    indexed_rows = np.empty(capacity, dtype=np.int32)
    cursor = 0
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row], matrix.indptr[row + 1]
        features = matrix.indices[start:stop]
        if not len(features):
            continue
        eligible = features[df[features] <= settings.max_probe_df]
        if not len(eligible):
            continue
        count = min(len(eligible), settings.indexed_grams)
        selected = eligible[np.argpartition(df[eligible], count - 1)[:count]]
        selected.sort()
        indexed_features[cursor:cursor + count] = selected
        indexed_rows[cursor:cursor + count] = row
        cursor += count
    indexed_features = indexed_features[:cursor]
    indexed_rows = indexed_rows[:cursor]
    counts = np.bincount(indexed_features, minlength=settings.hash_features)
    offsets = np.empty(settings.hash_features + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(indexed_features, kind="stable")
    posting_rows = indexed_rows[order]
    return FieldIndex(vectorizer, tfidf, matrix, common, offsets, posting_rows)


def _best_field_hits(index: FieldIndex, query_row: sparse.csr_matrix,
                     settings: Settings, k: int) -> np.ndarray:
    features = query_row.indices
    if not len(features) or k <= 0:
        return np.empty(0, dtype=np.int32)
    sizes = index.posting_offsets[features + 1] - index.posting_offsets[features]
    eligible = np.flatnonzero((sizes > 0) & (sizes <= settings.max_probe_df))
    if not len(eligible):
        return np.empty(0, dtype=np.int32)
    order = np.lexsort((features[eligible], sizes[eligible]))[:settings.query_grams]
    postings = []
    weights = []
    for position in eligible[order]:
        feature = features[position]
        start, stop = index.posting_offsets[feature:feature + 2]
        postings.append(index.posting_rows[start:stop])
        weights.append(np.full(stop - start, query_row.data[position], dtype=np.float32))
    rows, inverse = np.unique(np.concatenate(postings), return_inverse=True)
    votes = np.bincount(inverse, weights=np.concatenate(weights))
    ranked = np.lexsort((rows, -votes))[:k]
    return rows[ranked]


def _rank(name: float, address: float, channels: str) -> float:
    selected = set(channels.split("|"))
    return (0.55 * name + 0.45 * address + 0.12 * ("exact_name" in selected)
            + 0.08 * ("exact_address" in selected) + 0.02 * ("rare_name" in selected)
            + 0.02 * ("rare_address" in selected))


def _format_score(value: float) -> str:
    value = min(1.0, max(0.0, float(value)))
    if not math.isfinite(value):
        raise ValueError("non-finite cosine similarity")
    return f"{value:.6f}"


def _load_sorted_source1(path: Path, limit: int | None) -> list[Record]:
    return sorted(iter_records(path, 1, max_rows=limit), key=lambda record: record.entity_id)


def _build_lookups(names: list[str], addresses: list[str], settings: Settings):
    exact_names: dict[str, list[int]] = defaultdict(list)
    exact_addresses: dict[str, list[int]] = defaultdict(list)
    frequencies: Counter[str] = Counter()
    address_frequencies: Counter[str] = Counter()
    for idx, (name, address) in enumerate(zip(names, addresses, strict=True)):
        if name:
            exact_names[name].append(idx)
            frequencies.update(informative_name_tokens(name))
        if address:
            exact_addresses[address].append(idx)
            address_frequencies.update(informative_address_tokens(address))
    rare: dict[str, list[int]] = defaultdict(list)
    rare_address: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(names):
        for token in informative_name_tokens(name):
            if frequencies[token] <= settings.rare_max_df:
                rare[token].append(idx)
    for idx, address in enumerate(addresses):
        for token in informative_address_tokens(address):
            if address_frequencies[token] <= settings.rare_address_max_df:
                rare_address[token].append(idx)
    return exact_names, exact_addresses, rare, frequencies, rare_address, address_frequencies


def _part_path(work_dir: Path, source: int, chunk_number: int) -> Path:
    return work_dir / f"source{source}" / f"part_{chunk_number:06d}.tsv.gz"


def _read_part(path: Path) -> dict[str, list[tuple[str, str, str, str, str]]]:
    groups: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader)) != HEADER:
            raise ValueError(f"{path}: invalid staged header")
        for row in reader:
            if len(row) != 5:
                raise ValueError(f"{path}: invalid staged row")
            groups[row[0]].append(tuple(row))
    return groups


def _retrieve_source(
    source1: list[Record], target_path: Path, source: int, settings: Settings,
    work_dir: Path, limit_targets: int | None,
) -> None:
    total_chunks = (len(source1) + settings.query_chunk - 1) // settings.query_chunk
    missing = [i for i in range(total_chunks) if not _part_path(work_dir, source, i).is_file()]
    if not missing:
        return
    started = time.monotonic()
    targets = list(iter_records(target_path, source, max_rows=limit_targets))
    target_ids = [record.entity_id for record in targets]
    names = [normalize_name(record.business_name) for record in targets]
    addresses = [normalize_address(record.business_address) for record in targets]
    exact_names, exact_addresses, rare, frequencies, rare_address, address_frequencies = _build_lookups(names, addresses, settings)
    name_index = _field_index(names, settings)
    address_index = _field_index(addresses, settings)
    print(json.dumps({"stage": "target_index", "source": source, "targets": len(targets),
                      "seconds": round(time.monotonic() - started, 2)}), flush=True)
    del targets
    for chunk_number in missing:
        batch = source1[chunk_number * settings.query_chunk:(chunk_number + 1) * settings.query_chunk]
        query_names = [normalize_name(record.business_name) for record in batch]
        query_addresses = [normalize_address(record.business_address) for record in batch]
        name_query = name_index.query(query_names)
        address_query = address_index.query(query_addresses)
        part = _part_path(work_dir, source, chunk_number)
        part.parent.mkdir(parents=True, exist_ok=True)
        temp = part.with_suffix(part.suffix + ".tmp")
        with gzip.open(temp, "wt", encoding="utf-8", newline="", compresslevel=3) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(HEADER)
            for row_number, record in enumerate(batch):
                found: dict[int, set[str]] = defaultdict(set)
                name = query_names[row_number]
                address = query_addresses[row_number]
                if name and len(exact_names.get(name, ())) <= settings.max_exact_block:
                    for idx in exact_names.get(name, ()):
                        found[idx].add("exact_name")
                if address and len(exact_addresses.get(address, ())) <= settings.max_exact_block:
                    for idx in exact_addresses.get(address, ()):
                        found[idx].add("exact_address")
                tokens = informative_name_tokens(name)
                for token in sorted((t for t in tokens if t in rare), key=lambda t: (frequencies[t], t))[:3]:
                    for idx in rare[token]:
                        found[idx].add("rare_name")
                address_tokens = informative_address_tokens(address)
                for token in sorted((t for t in address_tokens if t in rare_address),
                                    key=lambda t: (address_frequencies[t], t))[:4]:
                    for idx in rare_address[token]:
                        found[idx].add("rare_address")
                for idx in _best_field_hits(name_index, name_query[row_number], settings, settings.name_k):
                    found[int(idx)].add("name_char")
                for idx in _best_field_hits(address_index, address_query[row_number], settings, settings.address_k):
                    found[int(idx)].add("address_char")
                if not found:
                    continue
                indices = np.fromiter(found, dtype=np.int32)
                name_scores = name_index.target[indices].dot(name_query[row_number].T).toarray().ravel()
                address_scores = address_index.target[indices].dot(address_query[row_number].T).toarray().ravel()
                candidates = []
                for pos, idx in enumerate(indices):
                    channels = "|".join(channel for channel in CHANNEL_ORDER if channel in found[int(idx)])
                    nscore = float(name_scores[pos])
                    ascore = float(address_scores[pos])
                    candidates.append((target_ids[int(idx)], _format_score(nscore), _format_score(ascore), channels,
                                       _rank(nscore, ascore, channels)))
                # Keep a generous per-source stage budget. Overall caps are applied in final assembly.
                candidates.sort(key=lambda row: (-row[4], row[0]))
                candidates = sorted(candidates[:max(64, settings.name_k + settings.address_k)], key=lambda row: row[0])
                for target_id, nscore, ascore, channels, _ in candidates:
                    writer.writerow((record.entity_id, target_id, nscore, ascore, channels))
        os.replace(temp, part)
        if chunk_number == missing[0] or (chunk_number + 1) % 20 == 0 or chunk_number == missing[-1]:
            print(json.dumps({"stage": "retrieval", "source": source, "chunk": chunk_number + 1,
                              "of": total_chunks, "seconds": round(time.monotonic() - started, 2)}), flush=True)


def _choose_rows(rows2: list, rows3: list, cap: int) -> list:
    def ordered(rows):
        return sorted(rows, key=lambda row: (-_rank(float(row[2]), float(row[3]), row[4]), row[1]))
    left, right = ordered(rows2), ordered(rows3)
    chosen = []
    quota = cap // 2
    chosen.extend(left[:quota])
    chosen.extend(right[:quota])
    remainder = left[quota:] + right[quota:]
    remainder.sort(key=lambda row: (-_rank(float(row[2]), float(row[3]), row[4]), row[1]))
    chosen.extend(remainder[:cap - len(chosen)])
    return sorted(chosen, key=lambda row: row[1])


def _manifest(data_root: Path, split: str, settings: Settings, limit_source1: int | None,
              limit_targets: int | None) -> dict:
    paths = [source_path(data_root, split, source) for source in (1, 2, 3)]
    retrieval_settings = {key: value for key, value in vars(settings).items() if key != "cap"}
    return {"retrieval_version": RETRIEVAL_VERSION, "split": split, "settings": retrieval_settings, "limit_source1": limit_source1,
            "limit_targets": limit_targets, "inputs": {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}}


def evaluate_retrieval(candidate_path: Path, truth_path: Path) -> dict:
    """Measure candidate coverage and its macro-F0.5 ceiling on a complete train set.

    This is an oracle ceiling, not the score of a trained matcher. It requires
    candidate rows for *every* training Source 1 record, including implicit
    zero-candidate rows. Do not use on `--limit-*` benchmark output.
    """
    truth = {source1: set(matches) for source1, matches in iter_truth(truth_path)}
    total_truth = len(truth)
    total_links = sum(len(matches) for matches in truth.values())
    all_counts: list[int] = []
    covered_links = 0
    fully_covered_all = 0
    fully_covered_nonempty = 0
    nonempty_truth = sum(bool(matches) for matches in truth.values())
    oracle_score = 0.0
    total_pairs = 0
    last_pair = ("", "")

    def finish(source1: str, candidates: set[str]) -> None:
        nonlocal covered_links, fully_covered_all, fully_covered_nonempty, oracle_score
        matches = truth.pop(source1, None)
        if matches is None:
            raise ValueError(f"candidate Source 1 ID absent from ground truth: {source1}")
        all_counts.append(len(candidates))
        hits = len(matches & candidates)
        covered_links += hits
        complete = hits == len(matches)
        fully_covered_all += complete
        fully_covered_nonempty += complete and bool(matches)
        if not matches:
            oracle_score += 1.0
        elif hits:
            recall = hits / len(matches)
            oracle_score += 1.25 * recall / (0.25 + recall)

    current_id = ""
    candidates: set[str] = set()
    with candidate_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if tuple(next(reader)) != HEADER:
            raise ValueError("candidate file has an invalid header")
        for row in reader:
            if len(row) != 5:
                raise ValueError("candidate file has an invalid row")
            source1, target = row[0], row[1]
            if (source1, target) <= last_pair:
                raise ValueError("candidate pairs are not unique and sorted")
            last_pair = (source1, target)
            if current_id and source1 != current_id:
                finish(current_id, candidates)
                candidates = set()
            current_id = source1
            candidates.add(target)
            total_pairs += 1
    if current_id:
        finish(current_id, candidates)
    for source1, matches in truth.items():
        all_counts.append(0)
        fully_covered_all += not matches
        oracle_score += not matches
    counts = np.asarray(all_counts, dtype=np.int32)
    return {"source1_rows": total_truth, "true_links": total_links, "candidate_pairs": total_pairs,
            "true_link_recall": covered_links / total_links if total_links else 1.0,
            "complete_set_coverage_all": fully_covered_all / total_truth if total_truth else 1.0,
            "complete_set_coverage_nonempty": fully_covered_nonempty / nonempty_truth if nonempty_truth else 1.0,
            "oracle_macro_f0_5": oracle_score / total_truth if total_truth else 1.0,
            "candidate_mean": float(counts.mean()) if len(counts) else 0.0,
            "candidate_p95": float(np.percentile(counts, 95)) if len(counts) else 0.0,
            "zero_candidate_rows": int(np.sum(counts == 0))}


def generate(data_root: Path, split: str, out: Path, work_dir: Path, settings: Settings,
             limit_source1: int | None = None, limit_targets: int | None = None) -> None:
    if settings.cap < 1 or settings.name_k < 0 or settings.address_k < 0 or settings.query_chunk < 1:
        raise ValueError("cap/query_chunk must be positive; top-K values must be nonnegative")
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    expected = _manifest(data_root, split, settings, limit_source1, limit_targets)
    if manifest_path.exists():
        found = json.loads(manifest_path.read_text(encoding="utf-8"))
        if found != expected:
            raise ValueError(f"{work_dir}: staged data has different inputs or settings; choose a fresh --work-dir")
    else:
        if any(work_dir.iterdir()):
            raise ValueError(f"{work_dir}: nonempty work directory has no manifest")
        manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    source1 = _load_sorted_source1(source_path(data_root, split, 1), limit_source1)
    for source in (2, 3):
        _retrieve_source(source1, source_path(data_root, split, source), source, settings, work_dir, limit_targets)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = out.with_suffix(out.suffix + ".tmp")
    emitted = 0
    with temp_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for chunk_number, start in enumerate(range(0, len(source1), settings.query_chunk)):
            groups2 = _read_part(_part_path(work_dir, 2, chunk_number))
            groups3 = _read_part(_part_path(work_dir, 3, chunk_number))
            for record in source1[start:start + settings.query_chunk]:
                rows = _choose_rows(groups2.get(record.entity_id, []), groups3.get(record.entity_id, []), settings.cap)
                for row in rows:
                    writer.writerow(row)
                    emitted += 1
    os.replace(temp_out, out)
    print(json.dumps({"stage": "complete", "source1": len(source1), "pairs": emitted,
                      "output": str(out), "work_dir": str(work_dir)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the final long-form candidate set")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, help="Persistent, private checkpoint directory")
    parser.add_argument("--name-k", type=int, default=32)
    parser.add_argument("--address-k", type=int, default=16)
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--matrix-chunk", type=int, default=100_000)
    parser.add_argument("--hash-features", type=int, default=1 << 20)
    parser.add_argument("--limit-source1", type=int, help="Benchmark only; not valid for final output")
    parser.add_argument("--limit-targets", type=int, help="Benchmark only; not valid for final output")
    parser.add_argument("--report", type=Path, help="Write train retrieval metrics after full generation")
    args = parser.parse_args()
    if args.report and (args.split != "train" or args.limit_source1 is not None or args.limit_targets is not None):
        parser.error("--report requires a complete, unbounded train run")
    settings = Settings(name_k=args.name_k, address_k=args.address_k, cap=args.cap,
                        query_chunk=args.query_chunk, matrix_chunk=args.matrix_chunk,
                        hash_features=args.hash_features)
    work_dir = args.work_dir or args.out.parent / f"{args.out.stem}.work"
    generate(args.data_root, args.split, args.out, work_dir, settings,
             args.limit_source1, args.limit_targets)
    if args.report:
        report = evaluate_retrieval(args.out, args.data_root / "train" / "train_ground_truth.tsv")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": "evaluation", **report}), flush=True)


if __name__ == "__main__":
    main()
