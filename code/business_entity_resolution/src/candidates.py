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
import multiprocessing
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

try:  # permit both `python -m src.candidates` and direct script execution
    from .data import Record, iter_records, iter_truth, sha256_file, source_path
    from .match_keys import KEY_NAMES, keys as match_keys
    from .normalization import (compact_name, informative_address_tokens, informative_name_tokens,
                                name_token_signature, normalize_address, normalize_name)
except ImportError:
    from data import Record, iter_records, iter_truth, sha256_file, source_path
    from match_keys import KEY_NAMES, keys as match_keys
    from normalization import (compact_name, informative_address_tokens, informative_name_tokens,
                               name_token_signature, normalize_address, normalize_name)

HEADER = ("source1_entity_id", "candidate_entity_id", "name_cosine", "address_cosine", "retrieval_channels")
CHANNEL_ORDER = ("exact_name", "exact_address", "name_signature", "compact_name", "rare_name", "rare_address", "name_pair", "address_pair", "cross_token", "single_name", "single_address", "name_char", "address_char", "sibling_name_char", "sibling_address_char", "second_hop", "reverse_top1") + KEY_NAMES
RETRIEVAL_VERSION = "name-key-1"


@dataclass(frozen=True)
class Settings:
    name_k: int = 32
    address_k: int = 16
    cap: int = 32
    final_score: str = "current"
    stage_cap: int = 64
    query_chunk: int = 2048
    matrix_chunk: int = 100_000
    hash_features: int = 1 << 20
    max_common_df: float = 0.05
    rare_max_df: int = 64
    rare_address_max_df: int = 128
    max_exact_block: int = 2_000
    indexed_grams: int = 8
    query_grams: int = 16
    gram_selection: str = "rarest"
    min_index_df: int = 1
    max_probe_df: int = 2_000
    cross_max_df: int = 1_000
    cross_name_tokens: int = 3
    cross_address_tokens: int = 4
    pair_max_df: int = 1_000
    pair_max_hits: int = 0
    single_max_df: int = 0
    single_tokens: int = 0
    name_keys: bool = False
    sibling_seeds: int = 0
    sibling_name_k: int = 16
    sibling_address_k: int = 8
    hop_seeds: int = 0
    hop_k: int = 16
    extra_pairs: Path | None = None
    key_cap: int = 0
    workers: int = 1
    config_hash: str | None = None


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


@dataclass
class TokenIndex:
    """Compact postings for informative whole tokens, including moderate-frequency ones."""

    vectorizer: HashingVectorizer
    posting_offsets: np.ndarray
    posting_rows: np.ndarray

    def probe(self, query_row: sparse.csr_matrix, limit: int, max_df: int) -> np.ndarray:
        features = query_row.indices
        if not len(features) or limit <= 0:
            return np.empty(0, dtype=np.int32)
        sizes = self.posting_offsets[features + 1] - self.posting_offsets[features]
        eligible = np.flatnonzero((sizes > 0) & (sizes <= max_df))
        if not len(eligible):
            return np.empty(0, dtype=np.int32)
        order = np.lexsort((features[eligible], sizes[eligible]))[:limit]
        postings = [self.posting_rows[self.posting_offsets[features[position]]:
                                     self.posting_offsets[features[position] + 1]]
                    for position in eligible[order]]
        return np.unique(np.concatenate(postings))

    def pair_probe(self, query_row: sparse.csr_matrix, limit: int, max_df: int,
                   max_hits: int = 0) -> np.ndarray:
        """Find targets sharing two informative query words in this field."""
        features = query_row.indices
        if len(features) < 2 or limit < 2:
            return np.empty(0, dtype=np.int32)
        sizes = self.posting_offsets[features + 1] - self.posting_offsets[features]
        eligible = np.flatnonzero((sizes > 0) & (sizes <= max_df))
        if len(eligible) < 2:
            return np.empty(0, dtype=np.int32)
        positions = eligible[np.lexsort((features[eligible], sizes[eligible]))[:limit]]
        postings = [self.posting_rows[self.posting_offsets[features[position]]:
                                     self.posting_offsets[features[position] + 1]]
                    for position in positions]
        intersections = [np.intersect1d(postings[i], postings[j], assume_unique=True)
                         for i in range(len(postings)) for j in range(i + 1, len(postings))]
        hits = [rows for rows in intersections if len(rows) and (not max_hits or len(rows) <= max_hits)]
        return np.unique(np.concatenate(hits)) if hits else np.empty(0, dtype=np.int32)


def _token_index(texts: list[str], settings: Settings, informative_tokens) -> TokenIndex:
    vectorizer = HashingVectorizer(
        analyzer="word", token_pattern=r"(?u)\b\w+\b", n_features=settings.hash_features,
        alternate_sign=False, norm=None, binary=True, dtype=np.float32,
    )
    blocks = []
    for start in range(0, len(texts), settings.matrix_chunk):
        normalized = [" ".join(informative_tokens(text))
                      for text in texts[start:start + settings.matrix_chunk]]
        blocks.append(vectorizer.transform(normalized))
    matrix = sparse.vstack(blocks, format="csr", dtype=np.float32)
    del blocks
    postings = matrix.tocsc()
    return TokenIndex(vectorizer, postings.indptr, postings.indices)


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
        eligible = features[(df[features] >= settings.min_index_df) &
                            (df[features] <= settings.max_probe_df)]
        if not len(eligible):
            continue
        count = min(len(eligible), settings.indexed_grams)
        if settings.gram_selection == "hash":
            # Stable bottom-k feature sampling avoids filling an index row
            # solely with typo-induced, document-frequency-one grams.
            priorities = eligible.astype(np.uint64) * np.uint64(11400714819323198485)
        else:
            priorities = df[eligible]
        selected = eligible[np.argpartition(priorities, count - 1)[:count]]
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
    if settings.gram_selection == "hash":
        priorities = features[eligible].astype(np.uint64) * np.uint64(11400714819323198485)
        order = np.argsort(priorities, kind="stable")[:settings.query_grams]
    else:
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
            + 0.08 * ("exact_address" in selected) + 0.08 * ("name_signature" in selected)
            + 0.04 * ("compact_name" in selected) + 0.02 * ("rare_name" in selected)
            + 0.02 * ("rare_address" in selected))


def _final_rank(name: float, address: float, channels: str, rule: str) -> float:
    if rule == "current":
        return _rank(name, address, channels)
    if rule == "balanced":
        return 0.5 * (name + address)
    raise ValueError(f"unknown final score rule: {rule}")


def _format_score(value: float) -> str:
    value = min(1.0, max(0.0, float(value)))
    if not math.isfinite(value):
        raise ValueError("non-finite cosine similarity")
    return f"{value:.6f}"


def _memory_state() -> dict[str, float]:
    """Report parent-process and, when available, container-wide memory."""
    state: dict[str, float] = {}
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", info.rss)
        if os.name == "posix":
            import resource
            # ru_maxrss is KiB on Linux and bytes on macOS.
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.uname().sysname == "Linux":
                peak *= 1024
        state.update({"rss_gb": round(info.rss / (1 << 30), 3),
                      "peak_rss_gb": round(peak / (1 << 30), 3)})
    except ImportError:
        pass
    for key, paths in {
        "container_current_gb": ("/sys/fs/cgroup/memory.current",
                                 "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        "container_peak_gb": ("/sys/fs/cgroup/memory.peak",
                              "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
    }.items():
        for path in paths:
            try:
                value = int(Path(path).read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                continue
            state[key] = round(value / (1 << 30), 3)
            break
    return state


def _load_sorted_source1(path: Path, limit: int | None) -> list[Record]:
    return sorted(iter_records(path, 1, max_rows=limit), key=lambda record: record.entity_id)


def _select_source1(source1: list[Record], sample_split: str | None,
                    shard_index: int = 0, shard_count: int = 1) -> list[Record]:
    """Select seeded evaluation rows or one sorted contiguous final-run shard."""
    if sample_split is None:
        start = len(source1) * shard_index // shard_count
        stop = len(source1) * (shard_index + 1) // shard_count
        return source1[start:stop]
    indices = random.Random(20260926).sample(range(len(source1)), min(60_000, len(source1)))
    if sample_split == "dev":
        chosen = indices[:10_000]
    elif sample_split == "holdout":
        chosen = indices[10_000:60_000]
    elif sample_split == "ranktrain":
        excluded = set(indices)
        available = [index for index in range(len(source1)) if index not in excluded]
        chosen = random.Random(7).sample(available, min(50_000, len(available)))
    else:
        raise ValueError(f"unknown sample split: {sample_split}")
    return [source1[index] for index in sorted(chosen)]


def _build_lookups(names: list[str], addresses: list[str], settings: Settings):
    exact_names: dict[str, list[int]] = defaultdict(list)
    exact_addresses: dict[str, list[int]] = defaultdict(list)
    name_signatures: dict[str, list[int]] = defaultdict(list)
    compact_names: dict[str, list[int]] = defaultdict(list)
    frequencies: Counter[str] = Counter()
    address_frequencies: Counter[str] = Counter()
    for idx, (name, address) in enumerate(zip(names, addresses, strict=True)):
        if name:
            exact_names[name].append(idx)
            frequencies.update(informative_name_tokens(name))
            if settings.name_keys:
                signature = name_token_signature(name)
                if signature:
                    name_signatures[signature].append(idx)
                compact = compact_name(name)
                if len(compact) >= 8 and (len(name.split()) == 1 or name.startswith("www ") or
                                           name.endswith((" com", " net", " org", " in"))):
                    compact_names[compact].append(idx)
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
    return (exact_names, exact_addresses, name_signatures, compact_names,
            rare, frequencies, rare_address, address_frequencies)


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


@dataclass
class _RetrievalState:
    source1: list[Record]
    target_ids: list[str]
    exact_names: dict[str, list[int]]
    exact_addresses: dict[str, list[int]]
    name_signatures: dict[str, list[int]]
    compact_names: dict[str, list[int]]
    rare_names: dict[str, list[int]]
    name_frequencies: Counter[str]
    rare_addresses: dict[str, list[int]]
    address_frequencies: Counter[str]
    name_index: FieldIndex
    address_index: FieldIndex
    name_tokens: TokenIndex
    address_tokens: TokenIndex
    extra: dict[str, tuple[int, ...]]
    match_blocks: dict[str, dict[str, list[int]]]
    settings: Settings
    work_dir: Path
    source: int


_FORK_STATE: _RetrievalState | None = None


def _always_staged(channels: str) -> bool:
    """Reverse and exact-key proposals bypass the _rank cut; the learned ranker decides them."""
    return "reverse_top1" in channels or "key_" in channels


def _match_blocks(targets: list[Record], cap: int) -> dict[str, dict[str, list[int]]]:
    if not cap:
        return {}
    raw = [match_keys(record.business_name, record.business_address) for record in targets]
    blocks: dict[str, dict[str, list[int]]] = {}
    for position, label in enumerate(KEY_NAMES):
        counts = Counter(key[position] for key in raw if key[position])
        block: dict[str, list[int]] = defaultdict(list)
        for idx, key in enumerate(raw):
            if key[position] and counts[key[position]] <= cap:
                block[key[position]].append(idx)
        blocks[label] = dict(block)
    return blocks


def _run_chunk(chunk_number: int) -> int:
    state = _FORK_STATE
    if state is None:
        raise RuntimeError("retrieval state was not initialized before worker fork")
    settings = state.settings
    batch = state.source1[chunk_number * settings.query_chunk:(chunk_number + 1) * settings.query_chunk]
    query_names = [normalize_name(record.business_name) for record in batch]
    query_addresses = [normalize_address(record.business_address) for record in batch]
    name_query = state.name_index.query(query_names)
    address_query = state.address_index.query(query_addresses)
    name_token_query = state.name_tokens.vectorizer.transform(
        [" ".join(informative_name_tokens(name)) for name in query_names]).tocsr()
    address_token_query = state.address_tokens.vectorizer.transform(
        [" ".join(informative_address_tokens(address)) for address in query_addresses]).tocsr()
    part = _part_path(state.work_dir, state.source, chunk_number)
    part.parent.mkdir(parents=True, exist_ok=True)
    temp = part.with_suffix(part.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", newline="", compresslevel=3) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for row_number, record in enumerate(batch):
            found: dict[int, set[str]] = defaultdict(set)
            name = query_names[row_number]
            address = query_addresses[row_number]
            if name and len(state.exact_names.get(name, ())) <= settings.max_exact_block:
                for idx in state.exact_names.get(name, ()):
                    found[idx].add("exact_name")
            if address and len(state.exact_addresses.get(address, ())) <= settings.max_exact_block:
                for idx in state.exact_addresses.get(address, ()):
                    found[idx].add("exact_address")
            if settings.name_keys and name:
                signature = name_token_signature(name)
                if signature and len(state.name_signatures.get(signature, ())) <= settings.max_exact_block:
                    for idx in state.name_signatures.get(signature, ()):
                        found[idx].add("name_signature")
                compact = compact_name(name)
                if len(compact) >= 8 and len(state.compact_names.get(compact, ())) <= settings.max_exact_block:
                    for idx in state.compact_names.get(compact, ()):
                        found[idx].add("compact_name")
            tokens = informative_name_tokens(name)
            for token in sorted((t for t in tokens if t in state.rare_names),
                                key=lambda t: (state.name_frequencies[t], t))[:3]:
                for idx in state.rare_names[token]:
                    found[idx].add("rare_name")
            address_tokens = informative_address_tokens(address)
            for token in sorted((t for t in address_tokens if t in state.rare_addresses),
                                key=lambda t: (state.address_frequencies[t], t))[:4]:
                for idx in state.rare_addresses[token]:
                    found[idx].add("rare_address")
            cross_name = state.name_tokens.probe(name_token_query[row_number],
                                                 settings.cross_name_tokens, settings.cross_max_df)
            cross_address = state.address_tokens.probe(address_token_query[row_number],
                                                       settings.cross_address_tokens, settings.cross_max_df)
            for idx in state.name_tokens.pair_probe(name_token_query[row_number],
                                                    settings.cross_name_tokens, settings.pair_max_df,
                                                    settings.pair_max_hits):
                found[int(idx)].add("name_pair")
            for idx in state.address_tokens.pair_probe(address_token_query[row_number],
                                                       settings.cross_address_tokens, settings.pair_max_df,
                                                       settings.pair_max_hits):
                found[int(idx)].add("address_pair")
            if len(cross_name) and len(cross_address):
                for idx in np.intersect1d(cross_name, cross_address, assume_unique=True):
                    found[int(idx)].add("cross_token")
            if settings.single_tokens and settings.single_max_df:
                for idx in state.name_tokens.probe(name_token_query[row_number],
                                                   settings.single_tokens, settings.single_max_df):
                    found[int(idx)].add("single_name")
                for idx in state.address_tokens.probe(address_token_query[row_number],
                                                      settings.single_tokens, settings.single_max_df):
                    found[int(idx)].add("single_address")
            for idx in _best_field_hits(state.name_index, name_query[row_number], settings, settings.name_k):
                found[int(idx)].add("name_char")
            for idx in _best_field_hits(state.address_index, address_query[row_number], settings, settings.address_k):
                found[int(idx)].add("address_char")
            for idx in state.extra.get(record.entity_id, ()):
                found[idx].add("reverse_top1")
            if state.match_blocks:
                for label, key in zip(KEY_NAMES, match_keys(record.business_name, record.business_address)):
                    for idx in state.match_blocks[label].get(key, ()) if key else ():
                        found[idx].add(label)
            if not found:
                continue
            if settings.sibling_seeds:
                initial_indices = np.fromiter(found, dtype=np.int32)
                initial_name = state.name_index.target[initial_indices].dot(name_query[row_number].T).toarray().ravel()
                initial_address = state.address_index.target[initial_indices].dot(address_query[row_number].T).toarray().ravel()
                seed_order = sorted(range(len(initial_indices)), key=lambda position: (
                    -_rank(float(initial_name[position]), float(initial_address[position]),
                           "|".join(channel for channel in CHANNEL_ORDER
                                    if channel in found[int(initial_indices[position])])),
                    state.target_ids[int(initial_indices[position])]))
                for position in seed_order[:settings.sibling_seeds]:
                    seed = int(initial_indices[position])
                    for idx in _best_field_hits(state.name_index, state.name_index.target[seed],
                                                settings, settings.sibling_name_k):
                        found[int(idx)].add("sibling_name_char")
                    for idx in _best_field_hits(state.address_index, state.address_index.target[seed],
                                                settings, settings.sibling_address_k):
                        found[int(idx)].add("sibling_address_char")
            if settings.hop_seeds:
                first = np.fromiter(found, dtype=np.int32)
                first_name = state.name_index.target[first].dot(name_query[row_number].T).toarray().ravel()
                first_address = state.address_index.target[first].dot(address_query[row_number].T).toarray().ravel()
                strong = np.flatnonzero((first_name >= 0.8) &
                                        ((first_address >= 0.6) | (first_name >= 0.95)))
                order = np.lexsort((first[strong],
                                    -(0.55 * first_name[strong] + 0.45 * first_address[strong])))
                for seed in first[strong[order[:settings.hop_seeds]]]:
                    for idx in _best_field_hits(state.name_index, state.name_index.target[int(seed)],
                                                settings, settings.hop_k):
                        found[int(idx)].add("second_hop")
                    for idx in _best_field_hits(state.address_index, state.address_index.target[int(seed)],
                                                settings, settings.hop_k // 2):
                        found[int(idx)].add("second_hop")
            indices = np.fromiter(found, dtype=np.int32)
            name_scores = state.name_index.target[indices].dot(name_query[row_number].T).toarray().ravel()
            address_scores = state.address_index.target[indices].dot(address_query[row_number].T).toarray().ravel()
            candidates = []
            for pos, idx in enumerate(indices):
                channels = "|".join(channel for channel in CHANNEL_ORDER if channel in found[int(idx)])
                nscore = float(name_scores[pos])
                ascore = float(address_scores[pos])
                candidates.append((state.target_ids[int(idx)], _format_score(nscore), _format_score(ascore), channels,
                                   _rank(nscore, ascore, channels)))
            # Keep the proposal shortlist separate from the final candidate cap.
            candidates.sort(key=lambda row: (-row[4], row[0]))
            keep = max(settings.stage_cap, settings.name_k + settings.address_k)
            candidates = sorted(candidates[:keep] +
                                [row for row in candidates[keep:] if _always_staged(row[3])],
                                key=lambda row: row[0])
            for target_id, nscore, ascore, channels, _ in candidates:
                writer.writerow((record.entity_id, target_id, nscore, ascore, channels))
    os.replace(temp, part)
    return chunk_number


def _load_extra(path: Path | None, target_ids: list[str],
                selected_source1: set[str], source: int) -> dict[str, tuple[int, ...]]:
    if path is None:
        return {}
    target_positions = {target_id: position for position, target_id in enumerate(target_ids)}
    extra: dict[str, set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", strict=True)
        if tuple(next(reader, ())) != ("source1_entity_id", "candidate_entity_id"):
            raise ValueError(f"{path}: invalid reverse-pair header")
        for line, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"{path}:{line}: expected two tab-separated IDs")
            source1_id, target_id = row
            if source1_id not in selected_source1 or not target_id.startswith(f"S{source}-"):
                continue
            position = target_positions.get(target_id)
            if position is None:
                raise ValueError(f"{path}:{line}: target ID absent from source {source}")
            extra[source1_id].add(position)
    return {source1_id: tuple(sorted(positions)) for source1_id, positions in extra.items()}


def _retrieve_source(
    source1: list[Record], target_path: Path, source: int, settings: Settings,
    work_dir: Path, limit_targets: int | None,
) -> None:
    global _FORK_STATE
    total_chunks = (len(source1) + settings.query_chunk - 1) // settings.query_chunk
    missing = [i for i in range(total_chunks) if not _part_path(work_dir, source, i).is_file()]
    if not missing:
        return
    started = time.monotonic()
    targets = list(iter_records(target_path, source, max_rows=limit_targets))
    target_ids = [record.entity_id for record in targets]
    names = [normalize_name(record.business_name) for record in targets]
    addresses = [normalize_address(record.business_address) for record in targets]
    (exact_names, exact_addresses, name_signatures, compact_names,
     rare, frequencies, rare_address, address_frequencies) = _build_lookups(names, addresses, settings)
    name_index = _field_index(names, settings)
    address_index = _field_index(addresses, settings)
    name_tokens = _token_index(names, settings, informative_name_tokens)
    address_tokens = _token_index(addresses, settings, informative_address_tokens)
    print(json.dumps({"stage": "target_index", "source": source, "targets": len(targets),
                      "seconds": round(time.monotonic() - started, 2), **_memory_state()}), flush=True)
    match_blocks = _match_blocks(targets, settings.key_cap)
    del targets, names, addresses
    extra = _load_extra(settings.extra_pairs, target_ids,
                        {record.entity_id for record in source1}, source)
    _FORK_STATE = _RetrievalState(source1, target_ids, exact_names, exact_addresses,
                                 name_signatures, compact_names, rare,
                                 frequencies, rare_address, address_frequencies,
                                 name_index, address_index, name_tokens, address_tokens, extra, match_blocks,
                                 settings, work_dir, source)
    if settings.workers > 1 and os.name == "posix":
        with multiprocessing.get_context("fork").Pool(processes=settings.workers) as pool:
            completed = pool.imap_unordered(_run_chunk, missing, chunksize=1)
            for count, chunk_number in enumerate(completed, start=1):
                if count == 1 or count % 20 == 0 or count == len(missing):
                    print(json.dumps({"stage": "retrieval", "source": source, "completed_chunks": count,
                                      "of": total_chunks, "last_chunk": chunk_number + 1,
                                      "seconds": round(time.monotonic() - started, 2),
                                      **_memory_state()}), flush=True)
    else:
        if settings.workers > 1:
            print(json.dumps({"stage": "warning", "message": "fork workers unavailable; using one process"}), flush=True)
        for count, chunk_number in enumerate(missing, start=1):
            _run_chunk(chunk_number)
            if count == 1 or count % 20 == 0 or count == len(missing):
                print(json.dumps({"stage": "retrieval", "source": source, "completed_chunks": count,
                                  "of": total_chunks, "last_chunk": chunk_number + 1,
                                  "seconds": round(time.monotonic() - started, 2),
                                  **_memory_state()}), flush=True)
    _FORK_STATE = None


def _choose_rows(rows2: list, rows3: list, cap: int, final_score: str = "current") -> list:
    def ordered(rows):
        return sorted(rows, key=lambda row: (-_final_rank(float(row[2]), float(row[3]), row[4], final_score), row[1]))
    left, right = ordered(rows2), ordered(rows3)
    chosen = []
    quota = cap // 2
    chosen.extend(left[:quota])
    chosen.extend(right[:quota])
    remainder = left[quota:] + right[quota:]
    remainder.sort(key=lambda row: (-_final_rank(float(row[2]), float(row[3]), row[4], final_score), row[1]))
    chosen.extend(remainder[:cap - len(chosen)])
    return sorted(chosen, key=lambda row: row[1])


def _manifest(data_root: Path, split: str, settings: Settings, limit_source1: int | None,
              limit_targets: int | None, sample_split: str | None,
              shard_index: int, shard_count: int) -> dict:
    paths = [source_path(data_root, split, source) for source in (1, 2, 3)]
    retrieval_settings = {key: value for key, value in vars(settings).items()
                          if key not in {"cap", "final_score", "workers", "extra_pairs", "config_hash"}}
    if settings.gram_selection == "hash":
        retrieval_settings["hash_query_order"] = "stable-v1"
    retrieval_settings["extra_pairs_sha256"] = (sha256_file(settings.extra_pairs)
                                                  if settings.extra_pairs else None)
    return {"retrieval_version": RETRIEVAL_VERSION, "split": split,
            "config_hash": settings.config_hash, "settings": retrieval_settings,
            "limit_source1": limit_source1, "sample_split": sample_split,
            "shard_index": shard_index, "shard_count": shard_count,
            "limit_targets": limit_targets, "inputs": {str(p.resolve()): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}}


def evaluate_retrieval(candidate_path: Path, truth_path: Path, selected_ids: set[str] | None = None) -> dict:
    """Measure candidate coverage and its macro-F0.5 ceiling on train rows.

    This is an oracle ceiling, not the score of a trained matcher. It requires
    candidate rows for every selected Source 1 record, including implicit
    zero-candidate rows. Do not use on `--limit-*` benchmark output.
    """
    truth = {source1: set(matches) for source1, matches in iter_truth(truth_path)
             if selected_ids is None or source1 in selected_ids}
    if selected_ids is not None and len(truth) != len(selected_ids):
        raise ValueError("selected Source 1 IDs absent from ground truth")
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
             limit_source1: int | None = None, limit_targets: int | None = None,
             sample_split: str | None = None, shard_index: int = 0,
             shard_count: int = 1) -> set[str] | None:
    if (settings.cap < 1 or settings.name_k < 0 or settings.address_k < 0 or
            settings.stage_cap < 1 or settings.query_chunk < 1 or settings.workers < 1 or
            settings.indexed_grams < 1 or settings.query_grams < 1 or
            settings.min_index_df < 1 or settings.max_probe_df < 1):
        raise ValueError("cap/chunk/workers/gram budgets must be positive; top-K values must be nonnegative")
    if settings.gram_selection not in {"rarest", "hash"}:
        raise ValueError("gram_selection must be rarest or hash")
    if settings.final_score not in {"current", "balanced"}:
        raise ValueError("final_score must be current or balanced")
    if settings.sibling_seeds < 0 or settings.sibling_name_k < 0 or settings.sibling_address_k < 0:
        raise ValueError("sibling seed and neighbor budgets must be nonnegative")
    if settings.key_cap < 0:
        raise ValueError("key_cap must be nonnegative")
    if settings.hop_seeds < 0 or settings.hop_k < 0:
        raise ValueError("hop seed and neighbor budgets must be nonnegative")
    if (settings.cross_max_df < 1 or settings.pair_max_df < 1 or settings.pair_max_hits < 0 or
            settings.single_max_df < 0 or settings.single_tokens < 0 or
            settings.cross_name_tokens < 0 or settings.cross_address_tokens < 0):
        raise ValueError("token frequency must be positive; optional token budgets must be nonnegative")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_count must be positive and 0 <= shard_index < shard_count")
    if sample_split and shard_count != 1:
        raise ValueError("sample split and final-run sharding cannot be combined")
    if settings.extra_pairs:
        metadata_path = settings.extra_pairs.with_suffix(settings.extra_pairs.suffix + ".meta.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("split") != split or
                metadata.get("source1_sha256") != sha256_file(source_path(data_root, split, 1)) or
                metadata.get("limit_targets") != limit_targets):
            raise ValueError("reverse pairs were built from a different split, Source 1 file, or target limit")
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    expected = _manifest(data_root, split, settings, limit_source1, limit_targets,
                         sample_split, shard_index, shard_count)
    if manifest_path.exists():
        found = json.loads(manifest_path.read_text(encoding="utf-8"))
        if found != expected:
            raise ValueError(f"{work_dir}: staged data has different inputs or settings; choose a fresh --work-dir")
    else:
        if any(work_dir.iterdir()):
            raise ValueError(f"{work_dir}: nonempty work directory has no manifest")
        manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    source1 = _select_source1(_load_sorted_source1(source_path(data_root, split, 1), limit_source1),
                              sample_split, shard_index, shard_count)
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
                rows = _choose_rows(groups2.get(record.entity_id, []), groups3.get(record.entity_id, []),
                                    settings.cap, settings.final_score)
                for row in rows:
                    writer.writerow(row)
                    emitted += 1
    os.replace(temp_out, out)
    print(json.dumps({"stage": "complete", "source1": len(source1), "pairs": emitted,
                      "output": str(out), "work_dir": str(work_dir)}), flush=True)
    return {record.entity_id for record in source1} if sample_split or shard_count > 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the final long-form candidate set")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, help="Persistent, private checkpoint directory")
    parser.add_argument("--name-k", type=int, default=32)
    parser.add_argument("--address-k", type=int, default=16)
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--final-score", choices=("current", "balanced"), default="current",
                        help="Final cap ranking; balanced uses equal name/address cosine without channel bonuses")
    parser.add_argument("--stage-cap", type=int, default=64,
                        help="Maximum ranked proposals staged per target source before the final cap")
    parser.add_argument("--query-chunk", type=int, default=2048)
    parser.add_argument("--matrix-chunk", type=int, default=100_000)
    parser.add_argument("--hash-features", type=int, default=1 << 20)
    parser.add_argument("--indexed-grams", type=int, default=8,
                        help="Maximum rare character grams posted per target field")
    parser.add_argument("--query-grams", type=int, default=16,
                        help="Maximum rare character grams probed per query field")
    parser.add_argument("--gram-selection", choices=("rarest", "hash"), default="rarest",
                        help="Choose target gram postings by rarest document frequency or stable hash")
    parser.add_argument("--min-index-df", type=int, default=1,
                        help="Minimum target frequency of indexed character grams")
    parser.add_argument("--max-probe-df", type=int, default=2_000,
                        help="Maximum target frequency of each probed character gram")
    parser.add_argument("--pair-max-df", type=int, default=1_000,
                        help="Maximum target frequency of a word used in same-field pair retrieval")
    parser.add_argument("--cross-max-df", type=int, default=1_000,
                        help="Maximum word frequency used in cross-field intersection proposals")
    parser.add_argument("--cross-name-tokens", type=int, default=3,
                        help="Informative query name words available to pair and cross-field probes")
    parser.add_argument("--cross-address-tokens", type=int, default=4,
                        help="Informative query address words available to pair and cross-field probes")
    parser.add_argument("--pair-max-hits", type=int, default=0,
                        help="Skip word-pair blocks larger than this; zero disables this limit")
    parser.add_argument("--single-max-df", type=int, default=0,
                        help="Maximum target frequency for single-word proposals; zero disables them")
    parser.add_argument("--single-tokens", type=int, default=0,
                        help="Rare query words to probe per field for single-word proposals")
    parser.add_argument("--name-keys", action="store_true",
                        help="Try word-order-independent and compact website-style name keys")
    parser.add_argument("--sibling-seeds", type=int, default=0,
                        help="Use this many top first-pass candidates as same-source second-hop queries")
    parser.add_argument("--sibling-name-k", type=int, default=16,
                        help="Name neighbors retrieved from each sibling seed")
    parser.add_argument("--sibling-address-k", type=int, default=8,
                        help="Address neighbors retrieved from each sibling seed")
    parser.add_argument("--hop-seeds", type=int, default=0,
                        help="Strong first-pass target rows to reuse as same-source queries")
    parser.add_argument("--hop-k", type=int, default=16,
                        help="Name neighbors per strong seed; address gets half this budget")
    parser.add_argument("--extra-pairs", type=Path,
                        help="Reverse top-1 TSV produced from Source 1 of this same split")
    parser.add_argument("--key-cap", type=int, default=0,
                        help="Maximum block size for exact phonetic/OCR/compact/number keys; zero disables them")
    parser.add_argument("--config-hash", help="SHA-256 of the canonical pipeline configuration")
    parser.add_argument("--workers", type=int, default=1, help="POSIX fork workers sharing the target index")
    parser.add_argument("--limit-source1", type=int, help="Benchmark only; not valid for final output")
    parser.add_argument("--limit-targets", type=int, help="Benchmark only; not valid for final output")
    parser.add_argument("--sample-split", choices=("dev", "holdout", "ranktrain"),
                        help="Seeded disjoint dev 10k, holdout 50k, or ranker-training 50k rows; train only")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based sorted Source 1 shard")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of contiguous Source 1 shards")
    parser.add_argument("--report", type=Path, help="Write train retrieval metrics after full generation")
    args = parser.parse_args()
    if args.sample_split and (args.split != "train" or args.limit_source1 is not None):
        parser.error("--sample-split requires train and cannot combine with --limit-source1")
    if args.report and (args.split != "train" or args.limit_source1 is not None or args.limit_targets is not None):
        parser.error("--report requires complete target sources and no --limit-* flags")
    settings = Settings(name_k=args.name_k, address_k=args.address_k, cap=args.cap,
                        final_score=args.final_score,
                        stage_cap=args.stage_cap,
                        query_chunk=args.query_chunk, matrix_chunk=args.matrix_chunk,
                        hash_features=args.hash_features, indexed_grams=args.indexed_grams,
                        query_grams=args.query_grams, gram_selection=args.gram_selection,
                        min_index_df=args.min_index_df, max_probe_df=args.max_probe_df,
                        pair_max_df=args.pair_max_df, pair_max_hits=args.pair_max_hits,
                        cross_max_df=args.cross_max_df,
                        cross_name_tokens=args.cross_name_tokens,
                        cross_address_tokens=args.cross_address_tokens,
                        single_max_df=args.single_max_df, single_tokens=args.single_tokens,
                        name_keys=args.name_keys, sibling_seeds=args.sibling_seeds,
                        sibling_name_k=args.sibling_name_k, sibling_address_k=args.sibling_address_k,
                        hop_seeds=args.hop_seeds, hop_k=args.hop_k,
                        extra_pairs=args.extra_pairs, key_cap=args.key_cap,
                        workers=args.workers, config_hash=args.config_hash)
    work_dir = args.work_dir or args.out.parent / f"{args.out.stem}.work"
    selected_ids = generate(args.data_root, args.split, args.out, work_dir, settings,
                            args.limit_source1, args.limit_targets, args.sample_split,
                            args.shard_index, args.shard_count)
    if args.report:
        report = evaluate_retrieval(args.out, args.data_root / "train" / "train_ground_truth.tsv", selected_ids)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": "evaluation", **report}), flush=True)


if __name__ == "__main__":
    main()
