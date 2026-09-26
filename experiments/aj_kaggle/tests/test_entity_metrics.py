"""Synthetic singleton-aware per-entity retrieval summaries."""

from pathlib import Path
import sys
import tempfile
import unittest

import duckdb
import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import diagnostics
from diagnostic_reporting import aggregate_entity_metrics, aggregate_metrics, entity_f05, entity_level_metrics
from statistical_diagnostics import stratified_bootstrap


class EntityMetricTests(unittest.TestCase):
    def test_stratified_bootstrap_is_paired_and_weights_country_macro_by_query_count(self):
        rows = {
            ("A", "legacy"): np.array([[0,0,0],[1,0,0]]),
            ("A", "preprocessed"): np.array([[0,0,0],[1,1,1]]),
            ("B", "legacy"): np.array([[1,1,1],[1,0,0],[0,0,0],[1,0,0]]),
            ("B", "preprocessed"): np.array([[1,1,1],[1,1,1],[0,0,0],[1,0,0]]),
        }
        result = stratified_bootstrap(rows, caps=(40,80), replicates=500, seed=7)
        overall = next(r for r in result if r["scope"] == "ALL_COUNTRIES_STRATIFIED"
                       and r["policy"] == "preprocessed" and r["cap"] == 40)
        self.assertAlmostEqual(overall["oracle_macro_f05"], 5/6)
        self.assertAlmostEqual(overall["micro_true_link_recall"], 3/4)
        self.assertAlmostEqual(overall["delta_oracle_f05_vs_legacy"], 1/3)
        self.assertEqual(overall["replicates"], 500)

    def test_candidate_label_shares_are_sample_composition_not_model_precision(self):
        row = aggregate_metrics({"metrics": [{
            "policy":"legacy", "cap":40, "true_links":10,
            "retained_links":8, "candidate_pairs":20,
            "lost_no_shared_key":1, "lost_frequency_gate":1,
            "lost_query_key_budget":0, "lost_rank_cap":0,
            "true_links_missing_address":2, "retained_missing_address_links":1,
        }]})[0]
        self.assertEqual(row["known_nonmatch_candidates"],12)
        self.assertAlmostEqual(row["candidate_pair_true_link_share"],0.4)
        self.assertAlmostEqual(row["candidate_pair_false_link_share"],0.6)

    def test_oracle_f05_includes_correct_singletons_and_reachable_recall(self):
        self.assertEqual(entity_f05(0, 0), 1.0)
        self.assertEqual(entity_f05(1, 0), 0.0)
        self.assertAlmostEqual(entity_f05(2, 1), 5/6)
        self.assertAlmostEqual(entity_f05(3, 1), 5/7)
        self.assertEqual(entity_f05(3, 3), 1.0)

    def test_entity_aggregates_distinguish_full_partial_miss_and_singleton(self):
        summary = aggregate_entity_metrics([
            {"qid":"singleton","true_links":0,"retained_links":0},
            {"qid":"complete","true_links":2,"retained_links":2},
            {"qid":"partial","true_links":3,"retained_links":1},
            {"qid":"miss","true_links":1,"retained_links":0},
        ])
        self.assertEqual(summary["queries"], 4)
        self.assertEqual(summary["singletons"], 1)
        self.assertEqual(summary["positive_queries"], 3)
        self.assertEqual(summary["complete_positive_queries"], 1)
        self.assertEqual(summary["partial_positive_queries"], 1)
        self.assertEqual(summary["positive_queries_with_zero_true_links_retrieved"], 1)
        self.assertAlmostEqual(summary["micro_true_link_recall"], 3/6)
        self.assertAlmostEqual(summary["macro_link_recall_positive_queries"], (1+1/3+0)/3)
        self.assertAlmostEqual(summary["median_link_recall_positive_queries"], 1/3)
        self.assertAlmostEqual(summary["complete_positive_query_rate"], 1/3)
        self.assertAlmostEqual(summary["positive_query_recall_coverage"], 2/3)
        self.assertAlmostEqual(summary["singleton_rate"], 1/4)
        self.assertAlmostEqual(summary["oracle_macro_f05_all_queries"], (1+1+5/7+0)/4)

    def test_entity_aggregates_reject_duplicate_queries_and_invalid_retrieval(self):
        with self.assertRaisesRegex(ValueError, "one combined row"):
            aggregate_entity_metrics([
                {"qid":"same","true_links":1,"retained_links":0},
                {"qid":"same","true_links":1,"retained_links":1},
            ])
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            aggregate_entity_metrics([{"qid":"bad","true_links":1,"retained_links":2}])
        with self.assertRaisesRegex(ValueError, "between zero"):
            entity_f05(1, 2)

    def test_sql_aggregates_source2_and_source3_before_entity_macro(self):
        country = "Syntheticland"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            db = duckdb.connect(str(output/"scratch.duckdb"))
            db.execute("CREATE TABLE sample(entity_id VARCHAR,country VARCHAR)")
            db.executemany("INSERT INTO sample VALUES (?,?)",[
                ("S1-single",country),("S1-complete",country),
                ("S1-partial",country),("S1-miss",country),
            ])
            db.execute("CREATE TABLE sampled_gt(source1_entity_id VARCHAR,matched_entity_ids VARCHAR)")
            db.executemany("INSERT INTO sampled_gt VALUES (?,?)",[
                ("S1-single",""),
                ("S1-complete","S2-a,S3-b"),
                ("S1-partial","S2-c,S2-d,S3-e"),
                ("S1-miss","S3-z"),
            ])
            db.close()

            fields = "qid VARCHAR,tid VARCHAR,shared BOOLEAN,usable BOOLEAN,selected BOOLEAN,rank INTEGER,target_address_empty BOOLEAN"
            per_source = {
                "source2": [
                    ("S1-complete","S2-a",True,True,True,39,False),
                    ("S1-partial","S2-c",True,True,True,2,False),
                    ("S1-partial","S2-d",True,True,True,41,False),
                ],
                "source3": [
                    ("S1-complete","S3-b",True,True,True,42,False),
                    ("S1-partial","S3-e",False,False,False,None,True),
                    ("S1-miss","S3-z",True,True,True,None,False),
                ],
            }
            for source,rows in per_source.items():
                path=output/f"stages_{diagnostics.part_name(country,source,'legacy')}.parquet"
                fixture=duckdb.connect()
                fixture.execute(f"CREATE TABLE stages({fields})")
                fixture.executemany("INSERT INTO stages VALUES (?,?,?,?,?,?,?)",rows)
                fixture.execute(f"COPY stages TO '{path.as_posix()}' (FORMAT parquet)")
                fixture.close()

            report={"sample":[{"country":country,"queries":4}],
                    "metrics":[{"policy":"legacy","cap":40}]}
            row=entity_level_metrics(report,output)[0]
            self.assertEqual(row["queries"],4)
            self.assertEqual(row["true_links"],6)
            self.assertEqual(row["retained_links"],2)
            self.assertEqual(row["singletons"],1)
            self.assertAlmostEqual(row["oracle_macro_f05_all_queries"],(1+5/6+5/7+0)/4)


if __name__ == "__main__":
    unittest.main()
