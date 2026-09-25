"""Contract tests for Person A's strict loader and retrieval handoff."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from candidates import HEADER, Settings, evaluate_retrieval, generate
from data import iter_records, iter_truth
from normalization import informative_address_tokens, informative_name_tokens, normalize_address, normalize_name


def write_tsv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class PersonATest(unittest.TestCase):
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
            generate(root, "test", output, root / "work", settings)
            self.assertEqual(output.read_bytes(), first)
            generate(root, "test", output, root / "work", Settings(**{**vars(settings), "cap": 1}))
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


if __name__ == "__main__":
    unittest.main()
