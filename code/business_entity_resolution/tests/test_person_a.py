"""Contract tests for Person A's strict loader and retrieval handoff."""

from __future__ import annotations

import csv
import gzip
import json
import pickle
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from candidates import (HEADER, FieldIndex, Settings, _best_field_hits,
                        _choose_rows, _select_source1, evaluate_retrieval, generate)
from assemble_staged import assemble
from aggregate_shard_reports import aggregate
from data import Record, iter_records, iter_truth
from merge_candidate_shards import merge
from recap_candidates import recap
from rank_candidates import (FEATURE_NAMES, FeatureContext, _retrieval_signature,
                             features_for_group, infer as rank_infer, select_ranked)
from reverse_top1 import run as run_reverse_top1
from stage_cutoff_curve import curve
from validate_candidate_long import validate
from normalization import informative_address_tokens, informative_name_tokens, normalize_address, normalize_name


def write_tsv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class NameScoreModel:
    def predict_proba(self, features):
        positive = features[:, FEATURE_NAMES.index("name_cosine")]
        return np.column_stack((1 - positive, positive))


class PersonATest(unittest.TestCase):
    def test_hash_query_probe_uses_same_priority_as_target_index(self):
        query = sparse.csr_matrix((np.ones(3, dtype=np.float32),
                                   ([0, 0, 0], [1, 2, 3])), shape=(1, 4))
        index = FieldIndex(None, None, None, None,
                           np.array([0, 0, 1, 2, 3], dtype=np.int64),
                           np.array([0, 1, 2], dtype=np.int32))
        settings = Settings(query_grams=1, max_probe_df=3)
        self.assertEqual(_best_field_hits(index, query, settings, 1).tolist(), [0])
        hashed = Settings(**{**vars(settings), "gram_selection": "hash"})
        self.assertEqual(_best_field_hits(index, query, hashed, 1).tolist(), [1])

    def test_cross_token_channel_recovers_name_and_address_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns,
                      [("S1-1", "Grand Chemical Systems", "529 Sunset Drive, Pullman", "US")])
            write_tsv(root / "test" / "test_source2.tsv", columns,
                      [("S2-1", "Grand Chemical Sytimes", "529 Sunset Dr, Pullamn", "US")])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Unrelated Business", "17 Other Avenue", "US")])
            output = root / "pairs.tsv"
            settings = Settings(name_k=0, address_k=0, rare_max_df=0,
                                rare_address_max_df=0, hash_features=1 << 12,
                                matrix_chunk=2, query_chunk=2)
            generate(root, "test", output, root / "work", settings)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual([(row[0], row[1]) for row in rows[1:]], [("S1-1", "S2-1")])
            self.assertIn("cross_token", rows[1][4].split("|"))

    def test_word_pairs_recover_name_only_and_address_only_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns, [
                ("S1-1", "Orthopedic Classic Health", "10405 Quince, Tucson", "US"),
                ("S1-2", "Different Trade Name", "4 Oakbriar Court, Penfield", "US"),
            ])
            write_tsv(root / "test" / "test_source2.tsv", columns, [
                ("S2-1", "Orthopedic Classic", "", "US"),
                ("S2-2", "Unrelated Brand", "4 Oakbriar Street, Penfield", "US"),
            ])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Another Business", "17 Other Avenue", "US")])
            output = root / "pairs.tsv"
            settings = Settings(name_k=0, address_k=0, rare_max_df=0,
                                rare_address_max_df=0, hash_features=1 << 12,
                                matrix_chunk=2, query_chunk=2)
            generate(root, "test", output, root / "work", settings)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))[1:]
            channels = {(row[0], row[1]): row[4].split("|") for row in rows}
            self.assertIn("name_pair", channels[("S1-1", "S2-1")])
            self.assertIn("address_pair", channels[("S1-2", "S2-2")])

    def test_single_word_channel_recovers_one_shared_distinctive_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns,
                      [("S1-1", "Orthopedic Group", "10405 Quince, Tucson", "US")])
            write_tsv(root / "test" / "test_source2.tsv", columns,
                      [("S2-1", "Orthopedic Clinic", "", "US")])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Another Business", "17 Other Avenue", "US")])
            output = root / "pairs.tsv"
            settings = Settings(name_k=0, address_k=0, rare_max_df=0,
                                rare_address_max_df=0, single_max_df=10,
                                single_tokens=1, hash_features=1 << 12,
                                matrix_chunk=2, query_chunk=2)
            generate(root, "test", output, root / "work", settings)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))[1:]
            self.assertEqual([(row[0], row[1]) for row in rows], [("S1-1", "S2-1")])
            self.assertEqual(rows[0][4], "single_name")

    def test_name_keys_recover_reordered_and_website_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns, [
                ("S1-1", "Golden Pub Clinic", "", "US"),
                ("S1-2", "Allied Family Practice", "", "US"),
            ])
            write_tsv(root / "test" / "test_source2.tsv", columns, [
                ("S2-1", "Clinic Golden Pub", "", "US"),
                ("S2-2", "alliedfamilypractice.com", "", "US"),
            ])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Unrelated", "", "US")])
            output = root / "pairs.tsv"
            settings = Settings(name_k=0, address_k=0, rare_max_df=0,
                                rare_address_max_df=0, name_keys=True,
                                hash_features=1 << 12, matrix_chunk=2, query_chunk=2)
            generate(root, "test", output, root / "work", settings)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))[1:]
            channels = {(row[0], row[1]): row[4].split("|") for row in rows}
            self.assertIn("name_signature", channels[("S1-1", "S2-1")])
            self.assertIn("compact_name", channels[("S1-2", "S2-2")])

    def test_hash_selected_grams_recover_a_typo_without_word_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns,
                      [("S1-1", "Grand Chemical Systems", "", "US")])
            write_tsv(root / "test" / "test_source2.tsv", columns,
                      [("S2-1", "Grand Chemical Sytimes", "", "US")])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Unrelated Industry", "", "US")])
            output = root / "pairs.tsv"
            settings = Settings(name_k=2, address_k=0, rare_max_df=0,
                                rare_address_max_df=0, cross_name_tokens=0,
                                cross_address_tokens=0, gram_selection="hash",
                                indexed_grams=16, query_grams=32,
                                hash_features=1 << 12, matrix_chunk=2, query_chunk=2)
            generate(root, "test", output, root / "work", settings)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))[1:]
            self.assertIn(("S1-1", "S2-1"), [(row[0], row[1]) for row in rows])
            self.assertIn("name_char", rows[0][4].split("|"))

    def test_second_hop_can_retrieve_neighbor_of_first_pass_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns,
                      [("S1-1", "Grand Chemical Systems", "", "US")])
            write_tsv(root / "test" / "test_source2.tsv", columns, [
                ("S2-1", "Grand Chemical Systems", "", "US"),
                ("S2-2", "Grand Chemical Sytimes", "", "US"),
            ])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Unrelated Industry", "", "US")])
            base = Settings(name_k=1, address_k=0, cap=2, stage_cap=2,
                            rare_max_df=0, rare_address_max_df=0, pair_max_df=1,
                            cross_name_tokens=0, cross_address_tokens=0,
                            indexed_grams=16, query_grams=32, min_index_df=2,
                            hash_features=1 << 12,
                            matrix_chunk=2, query_chunk=2)
            plain, expanded = root / "plain.tsv", root / "expanded.tsv"
            generate(root, "test", plain, root / "work-plain", base)
            generate(root, "test", expanded, root / "work-expanded",
                     Settings(**{**vars(base), "sibling_seeds": 1, "sibling_name_k": 2,
                                 "sibling_address_k": 0}))
            def read(path):
                with path.open("r", encoding="utf-8", newline="") as handle:
                    return {row[1]: row[4].split("|") for row in list(csv.reader(handle, delimiter="\t"))[1:]}
            self.assertNotIn("S2-2", read(plain))
            self.assertIn("sibling_name_char", read(expanded)["S2-2"])

    def test_strong_seed_hop_recovers_neighbor_without_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns,
                      [("S1-1", "Grand Chemical Systems", "", "US")])
            write_tsv(root / "test" / "test_source2.tsv", columns, [
                ("S2-1", "Grand Chemical Systems", "", "US"),
                ("S2-2", "Grand Chemical Sytimes", "", "US"),
            ])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Unrelated Industry", "", "US")])
            settings = Settings(name_k=1, address_k=0, cap=2, stage_cap=2,
                                rare_max_df=0, rare_address_max_df=0,
                                pair_max_df=1, cross_name_tokens=0,
                                cross_address_tokens=0, indexed_grams=16,
                                query_grams=32, min_index_df=2,
                                hash_features=1 << 12, matrix_chunk=2,
                                query_chunk=2, hop_seeds=1, hop_k=2)
            output = root / "pairs.tsv"
            generate(root, "test", output, root / "work", settings)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))[1:]
            channels = {row[1]: row[4].split("|") for row in rows}
            self.assertIn("second_hop", channels["S2-2"])

    def test_reverse_top1_pairs_survive_stage_and_reject_other_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns,
                      [("S1-TEST", "Alpha Supply", "1 Main Road", "US")])
            write_tsv(root / "test" / "test_source2.tsv", columns, [
                ("S2-1", "Alpha Supply", "1 Main Road", "US"),
                ("S2-2", "Alpha Other", "99 Other Lane", "US"),
            ])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Different Business", "Far Street", "US")])
            write_tsv(root / "train" / "train_source1.tsv", columns,
                      [("S1-TRAIN", "Alpha Supply", "1 Main Road", "US")])
            write_tsv(root / "train" / "train_source2.tsv", columns,
                      [("S2-TRAIN", "Alpha Supply", "1 Main Road", "US")])
            write_tsv(root / "train" / "train_source3.tsv", columns,
                      [("S3-TRAIN", "Different", "Elsewhere", "US")])
            reverse = root / "reverse.tsv"
            report = run_reverse_top1(root, "test", reverse, workers=1,
                                      batch_size=2, hash_features=1 << 12,
                                      matrix_chunk=2)
            self.assertEqual(report["split"], "test")
            with reverse.open("r", encoding="utf-8", newline="") as handle:
                reverse_rows = list(csv.reader(handle, delimiter="\t"))
            self.assertIn(["S1-TEST", "S2-2"], reverse_rows)
            settings = Settings(name_k=0, address_k=0, cap=3, stage_cap=1,
                                cross_name_tokens=0, cross_address_tokens=0,
                                hash_features=1 << 12, matrix_chunk=2,
                                query_chunk=2, extra_pairs=reverse)
            out = root / "pairs.tsv"
            generate(root, "test", out, root / "work", settings)
            with out.open("r", encoding="utf-8", newline="") as handle:
                channels = {row[1]: row[4].split("|")
                            for row in list(csv.reader(handle, delimiter="\t"))[1:]}
            self.assertIn("reverse_top1", channels["S2-2"])
            with gzip.open(root / "work" / "source2" / "part_000000.tsv.gz",
                           "rt", encoding="utf-8", newline="") as handle:
                staged = [row[1] for row in list(csv.reader(handle, delimiter="\t"))[1:]]
            self.assertIn("S2-2", staged)
            with self.assertRaisesRegex(ValueError, "different split"):
                generate(root, "train", root / "wrong.tsv", root / "wrong-work", settings)

    def test_stage_cutoff_curve_counts_ranked_links_before_final_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "train" / "train_source1.tsv", columns,
                      [("S1-1", "Example", "Address", "US")])
            write_tsv(root / "train" / "train_ground_truth.tsv",
                      ("source1_entity_id", "matched_entity_ids"), [("S1-1", "S2-2")])
            stage = root / "stage"
            for source, rows in ((2, [
                ("S1-1", "S2-1", "0.900000", "0.000000", "name_char"),
                ("S1-1", "S2-2", "0.800000", "0.000000", "name_char")]),
                                 (3, [])):
                part = stage / f"source{source}" / "part_000000.tsv.gz"
                part.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(part, "wt", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                    writer.writerow(HEADER)
                    writer.writerows(rows)
            report = curve(stage, root, "dev", [1, 2])
            self.assertEqual([item["covered_true_links"] for item in report["cutoffs_per_source"]], [0, 1])
            self.assertEqual(report["source_groups_with_candidates"], 1)
            self.assertEqual(report["source_groups_at_largest_cutoff"], 1)

    def test_cap_64_contains_candidates_needed_at_smaller_caps(self):
        rng = random.Random(24680)
        for case in range(100):
            left = [("S1-1", f"S2-{i:03d}", f"{rng.random():.6f}", f"{rng.random():.6f}", "name_char")
                    for i in range(rng.randrange(0, 90))]
            right = [("S1-1", f"S3-{i:03d}", f"{rng.random():.6f}", f"{rng.random():.6f}", "address_char")
                     for i in range(rng.randrange(0, 90))]
            large = {row[1] for row in _choose_rows(left, right, 64)}
            for cap in (16, 32):
                self.assertLessEqual({row[1] for row in _choose_rows(left, right, cap)}, large,
                                     f"cap nesting failed in case {case}")

    def test_balanced_final_score_can_undo_channel_bonus_displacement(self):
        rows = [
            ("S1-1", "S2-1", "0.700000", "0.700000", "exact_name"),
            ("S1-1", "S2-2", "0.760000", "0.760000", "name_char"),
        ]
        self.assertEqual(_choose_rows(rows, [], 1)[0][1], "S2-1")
        self.assertEqual(_choose_rows(rows, [], 1, "balanced")[0][1], "S2-2")

    def test_stage_cap_controls_proposal_shortlist_independently_of_final_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns,
                      [("S1-1", "Shared Name", "1 Main Road", "US")])
            write_tsv(root / "test" / "test_source2.tsv", columns, [
                ("S2-1", "Shared Name", "1 Main Road", "US"),
                ("S2-2", "Shared Name", "2 Main Road", "US"),
                ("S2-3", "Shared Name", "3 Main Road", "US"),
            ])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Unrelated", "Elsewhere", "US")])
            counts = []
            for stage_cap in (1, 3):
                output = root / f"pairs-{stage_cap}.tsv"
                settings = Settings(name_k=0, address_k=0, cap=3, stage_cap=stage_cap,
                                    rare_max_df=0, rare_address_max_df=0,
                                    hash_features=1 << 12, matrix_chunk=2, query_chunk=2)
                generate(root, "test", output, root / f"work-{stage_cap}", settings)
                with output.open("r", encoding="utf-8", newline="") as handle:
                    counts.append(len(list(csv.reader(handle, delimiter="\t"))) - 1)
            self.assertEqual(counts, [1, 3])

    def test_merge_shards_checks_global_pair_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.tsv", root / "second.tsv"
            write_tsv(first, HEADER, [("S1-1", "S2-1", "1", "1", "exact_name")])
            write_tsv(second, HEADER, [("S1-2", "S3-1", "1", "1", "exact_name")])
            self.assertEqual(merge([first, second], root / "merged.tsv"), 2)
            with self.assertRaisesRegex(ValueError, "order"):
                merge([second, first], root / "bad.tsv")

    def test_seeded_development_and_holdout_are_disjoint(self):
        records = [Record(f"S1-{i:05d}", "Name", "Address", "US") for i in range(12_000)]
        dev = _select_source1(records, "dev")
        holdout = _select_source1(records, "holdout")
        self.assertEqual(len(dev), 10_000)
        self.assertEqual(len(holdout), 2_000)
        self.assertFalse({record.entity_id for record in dev} & {record.entity_id for record in holdout})
        self.assertEqual([record.entity_id for record in dev], sorted(record.entity_id for record in dev))

    def test_ranker_training_sample_excludes_dev_and_holdout(self):
        records = [Record(f"S1-{i:05d}", "Name", "Address", "US") for i in range(110_000)]
        dev = {record.entity_id for record in _select_source1(records, "dev")}
        holdout = {record.entity_id for record in _select_source1(records, "holdout")}
        ranktrain = {record.entity_id for record in _select_source1(records, "ranktrain")}
        self.assertEqual((len(dev), len(holdout), len(ranktrain)), (10_000, 50_000, 50_000))
        self.assertFalse(dev & ranktrain or holdout & ranktrain)

    def test_ranker_features_and_per_source_quota(self):
        records = {
            "S1-1": Record("S1-1", "Alpha Supply", "94th Main Road", "US"),
            "S2-1": Record("S2-1", "Alpha Supply", "94nd Main Road", "US"),
            "S2-2": Record("S2-2", "Alpha Supply", "9 Other Road", "US"),
            "S3-1": Record("S3-1", "Alpha Supply", "94rd Main Road", "France"),
        }
        left = [("S1-1", "S2-1", "0.900000", "0.800000", "name_char"),
                ("S1-1", "S2-2", "0.850000", "0.100000", "name_char")]
        right = [("S1-1", "S3-1", "0.950000", "0.850000", "address_char")]
        rows, matrix = features_for_group("S1-1", left, right, FeatureContext(records))
        self.assertEqual(len(rows), 3)
        self.assertEqual(matrix.shape, (3, len(FEATURE_NAMES)))
        self.assertEqual(matrix[0, FEATURE_NAMES.index("num_any")], 1)
        self.assertEqual(matrix[0, FEATURE_NAMES.index("country_eq")], 1)
        self.assertEqual(matrix[2, FEATURE_NAMES.index("country_eq")], 0)
        self.assertEqual([row[1] for row in select_ranked(
            left, right, np.array([0.9, 0.8, 0.1]), 2)], ["S2-1", "S3-1"])
        self.assertEqual([row[1] for row in select_ranked(
            left, right, np.array([0.9, 0.8, 0.1]), 3)], ["S2-1", "S2-2", "S3-1"])

    def test_ranker_inference_reads_both_staged_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "test" / "test_source1.tsv", columns,
                      [("S1-1", "Alpha Supply", "94th Main Road", "US")])
            write_tsv(root / "test" / "test_source2.tsv", columns, [
                ("S2-1", "Alpha Supply", "94nd Main Road", "US"),
                ("S2-2", "Other Supply", "9 Other Road", "US")])
            write_tsv(root / "test" / "test_source3.tsv", columns,
                      [("S3-1", "Alpha Supply", "94rd Main Road", "France")])
            stage = root / "work"
            manifest = {"split": "test", "sample_split": None, "shard_index": 0,
                        "shard_count": 1, "limit_source1": None, "limit_targets": None,
                        "config_hash": "fixture-config",
                        "settings": {"stage_cap": 256, "query_chunk": 2,
                                     "extra_pairs_sha256": None}}
            stage.mkdir()
            (stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            for source, rows in ((2, [
                ("S1-1", "S2-1", "0.900000", "0.800000", "name_char"),
                ("S1-1", "S2-2", "0.700000", "0.100000", "name_char")]),
                                 (3, [("S1-1", "S3-1", "0.850000", "0.700000", "address_char")])):
                path = stage / f"source{source}" / "part_000000.tsv.gz"
                path.parent.mkdir()
                with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                    writer.writerow(HEADER)
                    writer.writerows(rows)
            model_path = root / "model.pkl"
            with model_path.open("wb") as handle:
                pickle.dump({"model": NameScoreModel(), "feature_names": FEATURE_NAMES,
                             "retrieval_signature": _retrieval_signature(manifest),
                             "config_hash": "fixture-config"}, handle)
            out = root / "ranked.tsv"
            report = rank_infer(root, "test", stage, model_path, out, 2)
            self.assertEqual(report["candidate_pairs"], 2)
            with out.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(tuple(rows[0]), HEADER)
            self.assertEqual([row[1] for row in rows[1:]], ["S2-1", "S3-1"])

            with model_path.open("wb") as handle:
                pickle.dump({"model": NameScoreModel(), "feature_names": FEATURE_NAMES,
                             "retrieval_signature": _retrieval_signature(manifest),
                             "config_hash": "other-config"}, handle)
            with self.assertRaisesRegex(ValueError, "config hashes"):
                rank_infer(root, "test", stage, model_path, root / "ranked-other.tsv", 2)

    def test_signature_ignores_split_specific_pair_files(self):
        def manifest(extra, blank):
            settings = {"stage_cap": 256, "query_chunk": 2, "extra_pairs_sha256": extra}
            if blank:
                settings["blank_pairs_sha256"] = blank
            return {"settings": settings}

        self.assertEqual(_retrieval_signature(manifest("train-top1", "train-blank")),
                         _retrieval_signature(manifest("test-top1", "test-blank")))
        self.assertNotEqual(_retrieval_signature(manifest("train-top1", "train-blank")),
                            _retrieval_signature(manifest("test-top1", None)))

    def test_sorted_shards_partition_every_source1_record(self):
        records = [Record(f"S1-{i:05d}", "Name", "Address", "US") for i in range(101)]
        shards = [_select_source1(records, None, index, 4) for index in range(4)]
        self.assertEqual([record for shard in shards for record in shard], records)
        self.assertEqual([len(shard) for shard in shards], [25, 25, 25, 26])

    def test_normalization_preserves_digits_and_avoids_blank_blocks(self):
        self.assertEqual(normalize_name("B+ Retail, Pvt. Ltd & Co"), "b retail private limited and company")
        self.assertEqual(normalize_address("12 St., Apt 4"), "12 street apartment 4")
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(informative_name_tokens("the private company"), ())
        self.assertIn("501", informative_address_tokens("apartment 501 mumbai"))
        self.assertEqual(normalize_name("École"), "ecole")
        self.assertIn("prjekts", normalize_name("আরবান প্রজেক্টস"))

    def test_loader_rejects_duplicate_and_bad_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.tsv"
            write_tsv(path, ("entity_id", "business_name", "business_address", "country"),
                      [("S1-1", "X", "Near A, Pune", "India"), ("S1-1", "Y", "Road B", "India")])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                list(iter_records(path, 1))
            write_tsv(path, ("entity_id", "business_name", "business_address", "country"),
                      [("S2-1", "X", "", "India")])
            with self.assertRaisesRegex(ValueError, "invalid source"):
                list(iter_records(path, 1))

    def test_truth_preserves_empty_list_and_rejects_repeated_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.tsv"
            write_tsv(path, ("source1_entity_id", "matched_entity_ids"),
                      [("S1-1", ""), ("S1-2", "S2-1,S3-2")])
            self.assertEqual(list(iter_truth(path)), [("S1-1", ()), ("S1-2", ("S2-1", "S3-2"))])
            write_tsv(path, ("source1_entity_id", "matched_entity_ids"), [("S1-1", "S2-1,S2-1")])
            with self.assertRaisesRegex(ValueError, "repeated"):
                list(iter_truth(path))

    def test_retrieval_contract_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "test"
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(split / "test_source1.tsv", columns, [
                ("S1-3", "École Atelier", "12 Rue de Paris, Lyon", "France"),
                ("S1-2", "", "", "US"),
                ("S1-1", "Orelee's Barbershop", "1795 Westchester Drive, High Point", "US"),
            ])
            write_tsv(split / "test_source2.tsv", columns, [
                ("S2-1", "Orelee's Barbershop", "1795 Westchester Dr, High Point", "US"),
                ("S2-2", "Different Salon", "42 Market Street", "US"),
            ])
            write_tsv(split / "test_source3.tsv", columns, [
                ("S3-1", "École Atelier", "12 Rue de Paris, Lyon", "France"),
                ("S3-2", "Some School", "1 Other Road", "France"),
            ])
            output = root / "pairs.tsv"
            settings = Settings(name_k=2, address_k=2, cap=2, query_chunk=2,
                                matrix_chunk=2, hash_features=1 << 12)
            generate(root, "test", output, root / "work", settings)
            first = output.read_bytes()
            self.assertEqual(assemble(root, "test", root / "work", root / "assembled.tsv", 2), 3)
            self.assertEqual((root / "assembled.tsv").read_bytes(), first)
            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(tuple(rows[0]), HEADER)
            self.assertTrue(all(len(row) == 5 for row in rows))
            self.assertTrue(all(row[0] != "S1-2" for row in rows[1:]))
            self.assertIn(("S1-1", "S2-1"), {(row[0], row[1]) for row in rows[1:]})
            self.assertIn(("S1-3", "S3-1"), {(row[0], row[1]) for row in rows[1:]})
            self.assertEqual([(row[0], row[1]) for row in rows[1:]],
                             sorted((row[0], row[1]) for row in rows[1:]))
            self.assertTrue(all(0 <= float(row[2]) <= 1 and 0 <= float(row[3]) <= 1 for row in rows[1:]))
            self.assertEqual(validate(output, root, "test", 2)["candidate_pairs"], 3)
            generate(root, "test", output, root / "work", settings)
            self.assertEqual(output.read_bytes(), first)
            generate(root, "test", output, root / "work", Settings(**{**vars(settings), "cap": 1}))
            full_cap = root / "full-cap.tsv"
            full_cap.write_bytes(first)
            recap(full_cap, root / "recapped.tsv", 1)
            self.assertEqual((root / "recapped.tsv").read_bytes(), output.read_bytes())
            with output.open("r", encoding="utf-8", newline="") as handle:
                capped = list(csv.reader(handle, delimiter="\t"))[1:]
            self.assertLessEqual(len(capped), 2)

    def test_oracle_metric_includes_singletons_and_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.tsv"
            pairs = root / "pairs.tsv"
            write_tsv(truth, ("source1_entity_id", "matched_entity_ids"),
                      [("S1-1", "S2-1,S3-1"), ("S1-2", ""), ("S1-3", "S2-3")])
            write_tsv(pairs, HEADER, [("S1-1", "S2-1", "0.8", "0.2", "name_char")])
            report = evaluate_retrieval(pairs, truth)
            self.assertEqual(report["source1_rows"], 3)
            self.assertEqual(report["true_links"], 3)
            self.assertAlmostEqual(report["true_link_recall"], 1 / 3)
            self.assertAlmostEqual(report["oracle_macro_f0_5"], (1.25 * 0.5 / 0.75 + 1) / 3)
            self.assertEqual(report["zero_candidate_rows"], 2)

    def test_aggregate_shard_reports_uses_link_and_entity_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            columns = ("entity_id", "business_name", "business_address", "country")
            write_tsv(root / "train" / "train_source1.tsv", columns,
                      [(f"S1-{index}", "Name", "Address", "US") for index in range(1, 5)])
            write_tsv(root / "train" / "train_ground_truth.tsv",
                      ("source1_entity_id", "matched_entity_ids"),
                      [("S1-1", "S2-1,S3-1"), ("S1-2", ""),
                       ("S1-3", "S2-3"), ("S1-4", "")])
            reports = [
                {"source1_rows": 2, "true_links": 2, "candidate_pairs": 3,
                 "true_link_recall": 0.5, "complete_set_coverage_all": 0.5,
                 "complete_set_coverage_nonempty": 0.0,
                 "oracle_macro_f0_5": (1.25 * 0.5 / 0.75 + 1) / 2,
                 "candidate_p95": 2, "zero_candidate_rows": 0},
                {"source1_rows": 2, "true_links": 1, "candidate_pairs": 3,
                 "true_link_recall": 1.0, "complete_set_coverage_all": 1.0,
                 "complete_set_coverage_nonempty": 1.0,
                 "oracle_macro_f0_5": 1.0, "candidate_p95": 2,
                 "zero_candidate_rows": 0},
            ]
            paths = [root / f"report-{index}.json" for index in range(2)]
            for path, report in zip(paths, reports, strict=True):
                path.write_text(json.dumps(report), encoding="utf-8")
            result = aggregate(root, paths, target_count=6, cap=2)
            self.assertEqual(result["covered_true_links"], 2)
            self.assertAlmostEqual(result["true_link_recall"], 2 / 3)
            self.assertAlmostEqual(result["complete_set_coverage_all"], 3 / 4)
            self.assertAlmostEqual(result["complete_set_coverage_nonempty"], 1 / 2)
            self.assertAlmostEqual(result["oracle_macro_f0_5"],
                                   ((1.25 * 0.5 / 0.75 + 1) + 2) / 4)
            self.assertEqual(result["candidate_p95"], 2)
            self.assertEqual(result["singleton_rows"], 2)
            self.assertAlmostEqual(result["pair_reduction_ratio"], 1 - 6 / 24)


if __name__ == "__main__":
    unittest.main()
