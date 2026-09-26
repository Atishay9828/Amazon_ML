"""Tests for cross-script, OCR, compact, and number proposal keys."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from candidates import Settings, generate
from match_keys import keys

COLUMNS = ("entity_id", "business_name", "business_address", "country")


def write_tsv(path: Path, rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)


class MatchKeyTest(unittest.TestCase):
    def test_indic_script_transliteration_shares_phonetic_key(self):
        for latin, indic in (("International Management Private Limited", "इंटरनेशनल मैनेजमेंट प्राइवेट लिमिटेड"),
                             ("Real Developers", "रियल डेवलपर्स"),
                             ("Unique Solutions Limited", "ಯುನೀಕ್ ಸೊಲ್ಯೂಷನ್ಸ್ ಲಿಮಿಟೆಡ್"),
                             ("International Surya Agro", "ఇంటర్నేషనల్ సూర్య ఆగ్రో")):
            self.assertEqual(keys(latin, "")[0], keys(indic, "")[0], latin)
            self.assertTrue(keys(latin, "")[0])

    def test_real_words_are_not_removed_as_legal_words(self):
        self.assertIn(" ", keys("Unique International Limited", "")[0])
        self.assertTrue(keys("Anand Constructions Private Limited", "")[0].startswith("knstrksns nd"))

    def test_ocr_key_repairs_digit_letter_swaps(self):
        self.assertEqual(keys("United C0mmittee", "")[1], keys("United Committee", "")[1])
        self.assertEqual(keys("K6D Associates", "")[1], keys("KGD Associates", "")[1])

    def test_compact_key_keeps_digits_and_drops_web_words(self):
        self.assertEqual(keys("Sector45bridge.Com", "")[2], keys("Sector 45 Bridge Limited", "")[2])
        self.assertEqual(keys("bangalore21network.com", "")[2], "bangalore21network")

    def test_number_key_strips_ordinals(self):
        self.assertEqual(keys("Stag", "4140 94th Street")[3], "4140 94|stg")
        self.assertEqual(keys("STAG", "4140- 94ND STREET, PO BOX")[3], "4140 94|stg")
        self.assertEqual(keys("Stag", "")[3], "")

    def test_empty_inputs_give_empty_keys(self):
        self.assertEqual(keys("", ""), ("", "", "", ""))


class KeyChannelTest(unittest.TestCase):
    def test_key_channel_proposes_transliterated_record_and_respects_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_tsv(root / "test" / "test_source1.tsv",
                      [("S1-1", "Real Developers", "Plot E-123, Jaipur", "India")])
            write_tsv(root / "test" / "test_source2.tsv",
                      [("S2-1", "रियल डेवलपर्स", "", "India"),
                       ("S2-2", "Other Name", "9 Elsewhere", "India")])
            write_tsv(root / "test" / "test_source3.tsv", [("S3-1", "Different", "1 Road", "US")])
            settings = Settings(name_k=0, address_k=0, key_cap=5, query_chunk=1)
            out = root / "pairs.tsv"
            generate(root, "test", out, root / "work", settings)
            with out.open("r", encoding="utf-8", newline="") as handle:
                rows = {(row[0], row[1]): row[4] for row in list(csv.reader(handle, delimiter="\t"))[1:]}
            self.assertIn("key_phonetic", rows[("S1-1", "S2-1")])

            capped = Settings(name_k=0, address_k=0, key_cap=0, query_chunk=1)
            generate(root, "test", root / "none.tsv", root / "work-none", capped)
            with (root / "none.tsv").open("r", encoding="utf-8", newline="") as handle:
                text = handle.read()
            self.assertNotIn("key_", text)


if __name__ == "__main__":
    unittest.main()
