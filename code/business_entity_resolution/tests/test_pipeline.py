"""Tests for the frozen pipeline configuration and the single-command runner."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
sys.path.insert(0, str(SRC))

from candidates import HEADER, Settings, generate
from pipeline_config import load_config
from run_pipeline import run as run_pipeline

COLUMNS = ("entity_id", "business_name", "business_address", "country")


def write_tsv(path: Path, rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def fixture_split(root: Path, split: str) -> None:
    write_tsv(root / split / f"{split}_source1.tsv", [
        ("S1-1", "Alpha Supply Private Limited", "12 Main Road, Pune", "India"),
        ("S1-2", "Beta Tools LLC", "7 Oak Street, Austin, TX", "US"),
        ("S1-3", "Gamma Boulangerie", "3 Rue de Paris, Lyon", "France")])
    write_tsv(root / split / f"{split}_source2.tsv", [
        ("S2-1", "Alpha Supply Pvt Ltd", "12 Main Rd, Pune", "India"),
        ("S2-2", "Beta Tools", "7 Oak St, Austin", "US"),
        ("S2-3", "Unrelated Name", "", "US")])
    write_tsv(root / split / f"{split}_source3.tsv", [
        ("S3-1", "Gamma Boulangerie SARL", "3 Rue de Paris, Lyon", "France"),
        ("S3-2", "Alpha Supply", "", "India")])


def write_config(path: Path, values: dict) -> Path:
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


class PipelineConfigTest(unittest.TestCase):
    def setUp(self):
        self.final = json.loads((CONFIGS / "final.json").read_text(encoding="utf-8"))
        self.baseline = json.loads((CONFIGS / "baseline.json").read_text(encoding="utf-8"))

    def test_committed_configs_load(self):
        self.assertEqual(len(load_config(CONFIGS / "final.json").sha256), 64)
        self.assertEqual(len(load_config(CONFIGS / "baseline.json").sha256), 64)

    def test_baseline_matches_candidate_defaults(self):
        defaults = vars(Settings())
        for key, value in self.baseline["candidates"].items():
            if key == "extra_pairs":
                self.assertIsNone(value)
            elif key != "workers":
                self.assertEqual(defaults[key], value, key)

    def test_hash_ignores_notes_and_key_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            noted = dict(self.final, _status="changed note", _anything={"x": 1})
            reordered = {key: self.final[key] for key in reversed(list(self.final))}
            first = load_config(write_config(root / "a.json", self.final)).sha256
            self.assertEqual(first, load_config(write_config(root / "b.json", noted)).sha256)
            self.assertEqual(first, load_config(write_config(root / "c.json", reordered)).sha256)

    def test_unknown_or_missing_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = json.loads(json.dumps(self.final))
            unknown["candidates"]["typo_setting"] = 1
            with self.assertRaisesRegex(ValueError, "unknown keys"):
                load_config(write_config(root / "unknown.json", unknown))
            missing = json.loads(json.dumps(self.final))
            del missing["candidates"]["stage_cap"]
            with self.assertRaisesRegex(ValueError, "missing keys"):
                load_config(write_config(root / "missing.json", missing))

    def test_two_sibling_mechanisms_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            both = json.loads(json.dumps(self.final))
            both["candidates"]["sibling_seeds"] = 2
            both["candidates"]["hop_seeds"] = 2
            with self.assertRaisesRegex(ValueError, "only one sibling"):
                load_config(write_config(Path(directory) / "both.json", both))

    def test_reverse_hash_depends_only_on_reverse_section(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed = json.loads(json.dumps(self.final))
            changed["candidates"]["stage_cap"] = 128
            a = load_config(write_config(root / "a.json", self.final))
            b = load_config(write_config(root / "b.json", changed))
            self.assertNotEqual(a.sha256, b.sha256)
            self.assertEqual(a.reverse_sha256, b.reverse_sha256)
            changed["reverse_top1"]["df_cap_fraction"] = 0.05
            c = load_config(write_config(root / "c.json", changed))
            self.assertNotEqual(a.reverse_sha256, c.reverse_sha256)


class RunPipelineTest(unittest.TestCase):
    def config_without_ranker(self, root: Path) -> Path:
        values = json.loads((CONFIGS / "final.json").read_text(encoding="utf-8"))
        values["ranker"] = {"enabled": False, "final_cap": 32}
        values["reverse_top1"]["workers"] = 1
        values["candidates"]["workers"] = 1
        values["candidates"]["query_chunk"] = 2
        return write_config(root / "config.json", values)

    def test_pipeline_matches_manual_steps_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            fixture_split(data, "test")
            config_path = self.config_without_ranker(root)
            out = root / "out" / "pairs.tsv"
            result = run_pipeline(config_path, data, "test", root / "work", out)
            config = load_config(config_path)
            self.assertEqual(result["config_hash"], config.sha256)
            reverse = root / "work" / "reverse-test.tsv"
            meta = json.loads(reverse.with_suffix(".tsv.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["config_hash"], config.reverse_sha256)

            blank = root / "work" / "reverse-blank-test.tsv"
            blank_meta = json.loads(blank.with_suffix(".tsv.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(blank_meta["channel"], "reverse_blank_name")
            self.assertEqual(blank_meta["blank_address_targets"], 2)

            manual_out = root / "manual.tsv"
            generate(data, "test", manual_out, root / "manual-work", config.settings(reverse, blank))
            self.assertEqual(out.read_text(encoding="utf-8"), manual_out.read_text(encoding="utf-8"))
            with out.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(tuple(rows[0]), HEADER)
            channels = {(row[0], row[1]): row[4] for row in rows[1:]}
            self.assertIn(("S1-1", "S2-1"), channels)
            self.assertIn("reverse_blank_name", channels[("S1-1", "S3-2")])
            with self.assertRaisesRegex(ValueError, "blank-address reverse pairs"):
                config.settings(reverse)

            again = run_pipeline(config_path, data, "test", root / "work", out)
            self.assertEqual(again["stage_dir"], result["stage_dir"])

    def test_reverse_file_from_other_settings_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            fixture_split(data, "test")
            config_path = self.config_without_ranker(root)
            run_pipeline(config_path, data, "test", root / "work", root / "pairs.tsv")
            values = json.loads(config_path.read_text(encoding="utf-8"))
            values["reverse_top1"]["df_cap_fraction"] = 0.5
            other = write_config(root / "other.json", values)
            with self.assertRaisesRegex(ValueError, "reverse metadata"):
                run_pipeline(other, data, "test", root / "work", root / "pairs-other.tsv")


if __name__ == "__main__":
    unittest.main()
