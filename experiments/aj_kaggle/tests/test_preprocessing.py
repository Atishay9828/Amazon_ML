"""Synthetic-only normalization checks; no challenge records are embedded."""

from __future__ import annotations

import unittest

import duckdb

from experiments.aj_kaggle.preprocessing import PREPROCESS_VERSION, normalized_sql


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = duckdb.connect()
        self.db.execute(
            "CREATE TABLE synthetic_records (entity_id VARCHAR, business_name VARCHAR, "
            "business_address VARCHAR, country VARCHAR)"
        )

    def tearDown(self) -> None:
        self.db.close()

    def normalize(self, name, address="", country="Exampleland", entity_id="S1-synthetic"):
        self.db.execute("DELETE FROM synthetic_records")
        self.db.execute(
            "INSERT INTO synthetic_records VALUES (?, ?, ?, ?)",
            [entity_id, name, address, country],
        )
        cursor = self.db.execute(normalized_sql("synthetic_records"))
        values = cursor.fetchone()
        return dict(zip([column[0] for column in cursor.description], values))

    def test_raw_columns_and_rid_are_retained_exactly(self):
        row = self.normalize("  ACME Ltd.  ", "12 Main Road", "Unknown Region X", " S1-01 ")
        self.assertEqual(row["business_name"], "  ACME Ltd.  ")
        self.assertEqual(row["business_address"], "12 Main Road")
        self.assertEqual(row["entity_id"], " S1-01 ")
        self.assertEqual(row["rid"], row["entity_id"])
        self.assertEqual(row["country"], "Unknown Region X")

    def test_whitespace_is_trimmed_and_collapsed(self):
        row = self.normalize(" \t ACME\n Widgets \r Ltd. \t", " \t 12  Main\nRoad,\r Suite 4 \t")
        self.assertEqual(row["name_folded"], "acme widgets ltd")
        self.assertEqual(row["name_core"], "acme widgets")
        self.assertEqual(row["address_folded"], "12 main road suite 4")
        self.assertEqual(row["address_canonical"], "12 main rd suite 4")

    def test_nulls_keep_raw_nulls_and_produce_empty_views(self):
        row = self.normalize(None, None, None, None)
        for field in ("entity_id", "business_name", "business_address", "country", "rid"):
            self.assertIsNone(row[field])
        for field in ("name_folded", "name_core", "address_folded", "address_canonical"):
            self.assertEqual(row[field], "")
        self.assertEqual(row["address_numbers"], [])
        self.assertTrue(row["name_is_empty"])
        self.assertTrue(row["name_core_is_empty"])
        self.assertTrue(row["address_is_empty"])

    def test_punctuation_only_fields_are_flagged_empty(self):
        row = self.normalize(" & / -- . ", " , # ( ) ")
        self.assertEqual(row["name_folded"], "")
        self.assertEqual(row["address_folded"], "")
        self.assertTrue(row["name_is_empty"])
        self.assertTrue(row["address_is_empty"])

    def test_repeated_legal_suffixes_are_removed_only_at_end(self):
        row = self.normalize("Acme Pvt. Ltd. Limited LLC")
        self.assertEqual(row["name_folded"], "acme pvt ltd limited llc")
        self.assertEqual(row["name_core"], "acme")
        row = self.normalize("The Ltd Studio Pvt Ltd")
        self.assertEqual(row["name_core"], "the ltd studio")
        row = self.normalize("Acme Pvt Ltd Studio")
        self.assertEqual(row["name_core"], "acme pvt ltd studio")

    def test_only_legal_suffixes_do_not_destroy_folded_view(self):
        row = self.normalize("Ltd. Pvt Limited")
        self.assertEqual(row["name_folded"], "ltd pvt limited")
        self.assertEqual(row["name_core"], "")
        self.assertFalse(row["name_is_empty"])
        self.assertTrue(row["name_core_is_empty"])

    def test_suffixes_require_whole_tokens_and_generic_words_remain(self):
        row = self.normalize("Incognito Ltdx Services Group Company")
        self.assertEqual(row["name_core"], "incognito ltdx services group company")
        row = self.normalize("Limited Edition Inc")
        self.assertEqual(row["name_core"], "limited edition")

    def test_ampersands_are_spaced_without_inventing_and(self):
        row = self.normalize("A&B and Sons")
        self.assertEqual(row["name_folded"], "a b and sons")
        self.assertNotEqual(row["name_folded"], "a and b and sons")

    def test_apostrophes_and_punctuation_separate_tokens(self):
        self.assertEqual(self.normalize("O'Reilly's")["name_folded"], "o reilly s")
        row = self.normalize("Alpha/Beta—Gamma, (R&D)")
        self.assertEqual(row["name_folded"], "alpha beta gamma r d")

    def test_address_abbreviations_are_an_additional_view(self):
        address = "1 Road Rd Street St Avenue Ave Boulevard Blvd Lane Ln Drive Dr"
        row = self.normalize("Synthetic", address)
        self.assertEqual(row["business_address"], address)
        self.assertEqual(row["address_folded"], address.lower())
        self.assertEqual(row["address_canonical"], "1 rd rd st st ave ave blvd blvd ln ln dr dr")

    def test_address_abbreviations_match_whole_tokens(self):
        row = self.normalize("Synthetic", "2 Broadway Streetlight Driveway, Rd.")
        self.assertEqual(row["address_canonical"], "2 broadway streetlight driveway rd")

    def test_all_number_runs_are_retained_without_assigning_semantics(self):
        row = self.normalize("Synthetic", "Unit 4B, 12-14 Main Road, Floor 2, 560001")
        self.assertEqual(row["address_numbers"], ["4", "12", "14", "2", "560001"])
        self.assertEqual(row["address_canonical"], "unit 4b 12 14 main rd floor 2 560001")
        self.assertFalse(any("house" in key or "first_number" in key for key in row))

    def test_number_order_repeats_and_leading_zeros_are_preserved(self):
        row = self.normalize("Synthetic", "Suite 2, 2 Main St 00002")
        self.assertEqual(row["address_numbers"], ["2", "2", "00002"])

    def test_single_digits_fraction_runs_and_postcode_leading_order(self):
        row = self.normalize("Synthetic", "00002, Unit 7, 1/2 Main Road")
        self.assertEqual(row["address_numbers"], ["00002", "7", "1", "2"])
        self.assertEqual(row["business_address"], "00002, Unit 7, 1/2 Main Road")
        self.assertEqual(row["address_canonical"], "00002 unit 7 1 2 main rd")

    def test_reordered_addresses_keep_their_order(self):
        first = self.normalize("Synthetic", "Unit 4, 12 Main Road, 00002")
        second = self.normalize("Synthetic", "00002, Main Road 12, Unit 4")
        self.assertEqual(first["address_numbers"], ["4", "12", "00002"])
        self.assertEqual(second["address_numbers"], ["00002", "12", "4"])
        self.assertNotEqual(first["address_canonical"], second["address_canonical"])

    def test_unicode_number_runs_are_preserved(self):
        row = self.normalize("Synthetic", "１２ Rue ٣، Apt ４B")
        self.assertEqual(row["address_numbers"], ["１２", "٣", "４"])

    def test_latin_accents_are_folded_after_nfc_normalization(self):
        row = self.normalize("CAFÉ Société Été Ltd", "12 Allée Émile Road")
        self.assertEqual(row["name_folded"], "cafe societe ete ltd")
        self.assertEqual(row["name_core"], "cafe societe ete")
        self.assertEqual(row["address_canonical"], "12 allee emile rd")
        row = self.normalize("Cafe\u0301 Ltd")
        self.assertEqual(row["name_core"], "cafe")

    def test_cyrillic_cjk_and_arabic_letters_survive(self):
        row = self.normalize("北京 Москва شركة", "東京 12 شارع")
        self.assertEqual(row["name_folded"], "北京 москва شركة")
        self.assertEqual(row["address_folded"], "東京 12 شارع")
        self.assertFalse(row["name_is_empty"])

    def test_devanagari_vowel_marks_survive(self):
        name = "श्री गणेश प्राइवेट लिमिटेड"
        address = "१२ महात्मा रोड, यूनिट ३"
        row = self.normalize(name, address)
        self.assertEqual(row["name_folded"], name)
        self.assertEqual(row["name_core"], name)
        self.assertEqual(row["address_folded"], "१२ महात्मा रोड यूनिट ३")
        self.assertEqual(row["address_numbers"], ["१२", "३"])

    def test_mixed_script_text_keeps_accents_and_combining_marks(self):
        row = self.normalize("Café श्री Industries Pvt Ltd", "12 Café सड़क Road")
        self.assertEqual(row["name_folded"], "café श्री industries pvt ltd")
        self.assertEqual(row["name_core"], "café श्री industries")
        self.assertEqual(row["address_canonical"], "12 café सड़क rd")

    def test_country_is_an_open_set_and_is_not_normalized(self):
        for country in ("France", "日本", "Côte d’Ivoire", "  Region-Ω  ", ""):
            with self.subTest(country=country):
                self.assertEqual(self.normalize("Synthetic", country=country)["country"], country)

    def test_folded_views_are_idempotent(self):
        for name, address in (
            (" Café ☕ Ltd. ", "12 Café Road, Unit 4"),
            ("Café श्री Pvt Ltd", "१२ सड़क Road"),
            ("A&B", " \t 2 -- Main Avenue . "),
            (None, None),
        ):
            with self.subTest(name=name):
                first = self.normalize(name, address)
                second = self.normalize(first["name_folded"], first["address_folded"])
                self.assertEqual(second["name_folded"], first["name_folded"])
                self.assertEqual(second["address_folded"], first["address_folded"])

    def test_core_and_canonical_views_are_idempotent(self):
        for name, address in (
            ("Acme Pvt Ltd", "Unit 4B, 12 Main Road 00002"),
            ("Café श्री Pvt Ltd", "१२ सड़क Road"),
            ("Ltd Pvt Limited", "1 Road Rd Street St"),
        ):
            with self.subTest(name=name):
                first = self.normalize(name, address)
                second = self.normalize(first["name_core"], first["address_canonical"])
                self.assertEqual(second["name_core"], first["name_core"])
                self.assertEqual(second["address_canonical"], first["address_canonical"])
                self.assertEqual(second["address_numbers"], first["address_numbers"])

    def test_version_and_relation_contract(self):
        self.assertIsInstance(PREPROCESS_VERSION, str)
        self.assertTrue(PREPROCESS_VERSION)
        with self.assertRaises(ValueError):
            normalized_sql(" \t ")
        with self.assertRaises(TypeError):
            normalized_sql(None)


if __name__ == "__main__":
    unittest.main()
