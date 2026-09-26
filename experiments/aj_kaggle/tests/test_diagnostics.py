"""Synthetic integration checks for diagnostic retrieval and loss attribution.

Run with the isolated DuckDB 1.3.2 runtime on sys.path. No package is installed
or downgraded here. Every source row, label, database, and report is invented and
written inside TemporaryDirectory; the real challenge files are never opened.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import duckdb

# diagnostics.py intentionally also runs as a standalone script and imports
# preprocessing directly. Supply that module directory for package test runs.
MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
import diagnostics


COUNTRY = "Exampleland (synthetic)"
SOURCE_HEADER = ("entity_id", "business_name", "business_address", "country")
TRUTH_HEADER = ("source1_entity_id", "matched_entity_ids")
TEST_CAPS = (40, 80, 160, 500)
LOSS_COLUMNS = (
    "lost_no_shared_key", "lost_frequency_gate", "lost_query_key_budget",
    "lost_rank_cap", "retained_links",
)


class CandidatePolicyContractTests(unittest.TestCase):
    def test_controlled_ablation_changes_one_retrieval_knob_at_a_time(self):
        policies = {row["name"]: row for row in diagnostics.POLICIES}
        self.assertEqual(len(policies), len(diagnostics.POLICIES))
        expected = {
            "preprocessed": (64, 3),
            "preprocessed_df256_k3": (256, 3),
            "preprocessed_df64_k6": (64, 6),
            "preprocessed_wider": (256, 6),
        }
        for name, values in expected.items():
            with self.subTest(policy=name):
                self.assertEqual(policies[name]["view"], "new")
                self.assertEqual((policies[name]["token_df"], policies[name]["keys"]), values)
        baseline = expected["preprocessed"]
        self.assertEqual(sum(a != b for a, b in zip(baseline, expected["preprocessed_df256_k3"])), 1)
        self.assertEqual(sum(a != b for a, b in zip(baseline, expected["preprocessed_df64_k6"])), 1)


def _write_tsv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _rows_as_dicts(cursor):
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


@unittest.skipUnless(duckdb.__version__ == "1.3.2", "requires isolated DuckDB 1.3.2 split semantics")
class DiagnosticsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="aj-diagnostics-synthetic-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        with duckdb.connect() as db:
            buckets = db.execute("""
                SELECT 'S1-synthetic-' || lpad(i::VARCHAR, 5, '0') AS rid,
                       hash(rid) % 100 AS bucket
                FROM range(2000) r(i) ORDER BY i
            """).fetchall()
        dev = [rid for rid, bucket in buckets if 5 <= bucket < 15][:7]
        cls.validation_ids = [rid for rid, bucket in buckets if bucket < 5][:2]
        cls.outside_id = next(rid for rid, bucket in buckets if bucket >= 15)
        cls.qids = dict(zip(("df", "dedup", "budget", "no_shared", "rank", "one_true", "singleton"), dev))
        names = {
            "df": "commonkey queryphrase",
            "dedup": "repeatkey queryphrase",
            "budget": "alpha beta gamma zeta",
            "no_shared": "querypebble",
            "rank": "rankmesh",
            "one_true": "onlyonecue",
            "singleton": "singletonmesh",
        }
        cls.sources = {1: [], 2: [], 3: []}
        cls.truth = {rid: [] for rid in dev + cls.validation_ids + [cls.outside_id]}
        for kind, rid in cls.qids.items():
            cls.sources[1].append((rid, names[kind], "", COUNTRY))
        for rid in cls.validation_ids + [cls.outside_id]:
            cls.sources[1].append((rid, "excluded synthetic query", "", COUNTRY))

        for source in (2, 3):
            prefix = f"S{source}-"

            def add(target_id, name):
                cls.sources[source].append((target_id, name, "", COUNTRY))

            # All 65 records count towards DF, although only one is labeled.
            # Repeated tokens within each row must still count only once.
            for i in range(65):
                add(f"{prefix}df-{i:04d}", "commonkey commonkey targetphrase")
            cls.truth[cls.qids["df"]].append(f"{prefix}df-0000")
            # Exactly at the 64-document cutoff. Counting token occurrences
            # instead of distinct records would incorrectly discard this key.
            for i in range(64):
                add(f"{prefix}dedup-{i:04d}", "repeatkey repeatkey targetphrase")
            cls.truth[cls.qids["dedup"]].append(f"{prefix}dedup-0000")
            # Equal DF: lexical alpha/beta/gamma displace the true zeta key
            # under the three-key budget. The wider policy recovers it.
            for token in ("alpha", "beta", "gamma", "zeta"):
                add(f"{prefix}budget-{token}", f"{token} alienpayload")
            cls.truth[cls.qids["budget"]].append(f"{prefix}budget-zeta")
            add(f"{prefix}no-shared", "targetgranite")
            cls.truth[cls.qids["no_shared"]].append(f"{prefix}no-shared")
            # Identical ranking evidence makes target ID the decisive tie-break.
            for i in range(161):
                add(f"{prefix}rank-{i:04d}", "rankmesh")
            indices = (39, 40, 79, 80, 159, 160) if source == 2 else (39, 79, 159)
            cls.truth[cls.qids["rank"]].extend(f"{prefix}rank-{i:04d}" for i in indices)
            # This query has ONE true link overall. Five hundred equally ranked
            # distractors precede it, so even cap=500 loses that single true link.
            for i in range(501):
                add(f"{prefix}one-{i:04d}", "onlyonecue")
            if source == 2:
                cls.truth[cls.qids["one_true"]].append(f"{prefix}one-0500")
            add(f"{prefix}singleton-distractor", "singletonmesh")

        cls.data = cls._write_fixture("baseline")
        cls.report, cls.captures = cls._run_capture(cls.data, cls.root / "baseline-output")

    @classmethod
    def _write_fixture(cls, name, truth=None):
        data = cls.root / name / "dataset"
        for source, rows in cls.sources.items():
            _write_tsv(data / "train" / f"train_source{source}.tsv", SOURCE_HEADER, rows)
        labels = cls.truth if truth is None else truth
        _write_tsv(
            data / "train" / "train_ground_truth.tsv", TRUTH_HEADER,
            ((rid, ",".join(targets)) for rid, targets in labels.items()),
        )
        return data

    @classmethod
    def _run_capture(cls, data, output, per_country=100):
        captures = {}
        original = diagnostics.run_policy

        def capture(audit, policy, country, source):
            result = original(audit, policy, country, source)
            captures[(source, policy["name"])] = {
                "frequencies": dict(audit.run("SELECT key,freq FROM frequencies ORDER BY key").fetchall()),
                "selected": audit.run("SELECT rid,key FROM selected ORDER BY rid,key").fetchall(),
                "ranked": audit.run("SELECT qid,tid,rank,rank_score,key_hits FROM ranked ORDER BY qid,rank").fetchall(),
                "stages": _rows_as_dicts(audit.run("SELECT * FROM stages ORDER BY qid,tid")),
                "target_batches": audit.run("SELECT count(DISTINCT diagnostic_batch) FROM normalized_targets").fetchone()[0],
            }
            return result

        # Test the 500 cutoff even when changing the exploratory default caps.
        with patch.object(diagnostics, "run_policy", side_effect=capture), patch.object(diagnostics, "CAPS", TEST_CAPS), patch.object(diagnostics, "TARGET_BATCH_ROWS", 32), redirect_stdout(io.StringIO()):
            report = diagnostics.run_diagnostics(data, output, per_country=per_country, memory_mb=128)
        return report, captures

    def _stage(self, source, policy, kind, target_suffix):
        target = f"S{source[-1]}-{target_suffix}"
        return next(
            row for row in self.captures[(source, policy)]["stages"]
            if row["qid"] == self.qids[kind] and row["tid"] == target
        )

    def test_frozen_sample_excludes_original_validation_and_outside_buckets(self):
        with duckdb.connect() as db:
            rows = db.execute("SELECT entity_id,original_bucket FROM read_parquet(?)", [str(self.root / "baseline-output" / "frozen_queries.parquet")]).fetchall()
        self.assertEqual({rid for rid, _ in rows}, set(self.qids.values()))
        self.assertTrue(all(5 <= bucket < 15 for _, bucket in rows))
        self.assertTrue(set(self.validation_ids + [self.outside_id]).isdisjoint(rid for rid, _ in rows))
        self.assertEqual(self.report["sample"][0]["country"], COUNTRY)
        self.assertEqual(self.report["manifest"]["duckdb"], "1.3.2")
        self.assertEqual(self.report['semantics_sha256'],diagnostics.semantics_sha256())
        self.assertEqual(self.report['sample_sha256'],diagnostics.sha256_file(self.root/'baseline-output'/'frozen_queries.parquet'))

    def test_small_sample_uses_stable_sha256_order_after_dev_exclusion(self):
        report, _ = self._run_capture(self.data, self.root / "small-output", per_country=3)
        expected = sorted(self.qids.values(), key=lambda rid: (hashlib.sha256(("aj-diagnostic-v1:" + rid).encode()).hexdigest(), rid))[:3]
        with duckdb.connect() as db:
            actual = db.execute("SELECT entity_id FROM read_parquet(?)", [str(self.root / "small-output" / "frozen_queries.parquet")]).fetchall()
        self.assertEqual({row[0] for row in actual}, set(expected))
        self.assertEqual(report["sample"][0]["queries"], 3)

    def test_full_target_df_includes_unlabeled_distractors_and_deduplicates_tokens(self):
        for source in ("source2", "source3"):
            for policy in (p["name"] for p in diagnostics.POLICIES):
                with self.subTest(source=source, policy=policy):
                    frequencies = self.captures[(source, policy)]["frequencies"]
                    self.assertGreater(self.captures[(source, policy)]["target_batches"], 1)
                    self.assertEqual(frequencies["NT=commonkey"], 65)
                    self.assertEqual(frequencies["NT=repeatkey"], 64)
                    df_stage = self._stage(source, policy, "df", "df-0000")
                    self.assertTrue(df_stage["shared"])
                    df_wide = policy in {"preprocessed_df256_k3", "preprocessed_wider"}
                    self.assertEqual(df_stage["usable"], df_wide)
                    self.assertEqual(df_stage["selected"], df_wide)
                    dedup_stage = self._stage(source, policy, "dedup", "dedup-0000")
                    self.assertTrue(dedup_stage["usable"])
                    self.assertTrue(dedup_stage["selected"])
                    self.assertEqual(dedup_stage["rank"], 1)

    def test_query_budget_loss_is_distinct_from_df_loss_and_wider_keys_recover(self):
        for source in ("source2", "source3"):
            for policy in (p["name"] for p in diagnostics.POLICIES):
                with self.subTest(source=source, policy=policy):
                    stage = self._stage(source, policy, "budget", "budget-zeta")
                    self.assertTrue(stage["shared"])
                    self.assertTrue(stage["usable"])
                    has_wider_budget = policy in {"preprocessed_df64_k6", "preprocessed_wider"}
                    self.assertEqual(stage["selected"], has_wider_budget)
                    keys = {key for qid, key in self.captures[(source, policy)]["selected"] if qid == self.qids["budget"]}
                    expected = {"NT=alpha", "NT=beta", "NT=gamma"}
                    if has_wider_budget:
                        expected.add("NT=zeta")
                    self.assertEqual(keys, expected)

    def test_ablation_policies_isolate_frequency_and_key_budget(self):
        policies = {row["name"]: row for row in diagnostics.POLICIES}
        baseline = policies["preprocessed"]
        df_only = policies["preprocessed_df256_k3"]
        budget_only = policies["preprocessed_df64_k6"]
        combined = policies["preprocessed_wider"]
        self.assertEqual((baseline["token_df"], baseline["keys"]), (64, 3))
        self.assertEqual((df_only["token_df"], df_only["keys"]), (256, 3))
        self.assertEqual((budget_only["token_df"], budget_only["keys"]), (64, 6))
        self.assertEqual((combined["token_df"], combined["keys"]), (256, 6))

    def test_budget_is_separate_for_each_token_channel_and_exact_keys(self):
        with duckdb.connect() as db:
            db.execute("CREATE TABLE qk(rid VARCHAR,key VARCHAR)")
            db.execute("CREATE TABLE frequencies(key VARCHAR,freq BIGINT)")
            keys = [f"{channel}{word}" for channel in ("NT=", "AT=", "AN=12:") for word in ("alpha", "beta", "gamma", "zeta")]
            keys += ["N=exact name", "A=exact address"]
            db.executemany("INSERT INTO qk VALUES ('S1-synthetic',?)", [(key,) for key in keys])
            db.executemany("INSERT INTO frequencies VALUES (?,1)", [(key,) for key in keys])
            selected = {row[1] for row in db.execute(diagnostics.selected_keys_sql(64, 3)).fetchall()}
        self.assertEqual(len(selected), 11)
        self.assertIn("N=exact name", selected)
        self.assertIn("A=exact address", selected)
        self.assertFalse(any("zeta" in key for key in selected))

    def test_rank_ties_use_target_ids_and_caps_do_not_force_in_true_matches(self):
        expected_indices = {"source2": (39, 40, 79, 80, 159, 160), "source3": (39, 79, 159)}
        for source in ("source2", "source3"):
            for policy in (p["name"] for p in diagnostics.POLICIES):
                rows = [row for row in self.captures[(source, policy)]["ranked"] if row[0] == self.qids["rank"]]
                self.assertEqual([row[1] for row in rows], sorted(row[1] for row in rows))
                self.assertEqual([row[2] for row in rows], list(range(1, 162)))
                self.assertEqual(len({row[3] for row in rows}), 1)
                positives = [row for row in self.captures[(source, policy)]["stages"] if row["qid"] == self.qids["rank"]]
                self.assertEqual(sorted(row["rank"] for row in positives), [i + 1 for i in expected_indices[source]])
                for cap in TEST_CAPS:
                    self.assertEqual(sum(row["rank"] <= cap for row in positives), sum(i < cap for i in expected_indices[source]))

    def test_one_true_link_can_be_lost_even_at_cap_500(self):
        self.assertEqual(len(self.truth[self.qids["one_true"]]), 1)
        for policy in (p["name"] for p in diagnostics.POLICIES):
            stage = self._stage("source2", policy, "one_true", "one-0500")
            self.assertTrue(stage["selected"])
            self.assertEqual(stage["rank"], 501)
            self.assertTrue(all(stage["rank"] > cap for cap in TEST_CAPS))

    def test_blank_address_and_no_name_key_remains_a_retrieval_miss(self):
        for source in ("source2", "source3"):
            for policy in (p["name"] for p in diagnostics.POLICIES):
                stage = self._stage(source, policy, "no_shared", "no-shared")
                self.assertFalse(stage["shared"])
                self.assertFalse(stage["usable"])
                self.assertFalse(stage["selected"])
                self.assertIsNone(stage["rank"])

    def test_loss_totals_reconcile_and_singleton_candidates_are_not_predictions(self):
        for metric in self.report["metrics"]:
            with self.subTest(source=metric["source"], policy=metric["policy"], cap=metric["cap"]):
                self.assertEqual(metric["true_links"], sum(metric[key] for key in LOSS_COLUMNS))
                self.assertEqual(metric["queries"], len(self.qids))
                self.assertEqual(metric["known_nonmatch_candidates"], metric["candidate_pairs"] - metric["retained_links"])
                self.assertLessEqual(metric["candidates_max"], metric["cap"])
                self.assertEqual(metric["true_links_missing_address"], metric["true_links"])
                self.assertEqual(metric["retained_missing_address_links"], metric["retained_links"])
                self.assertEqual(metric["missing_address_link_recall"], metric["candidate_link_recall"])
        for capture in self.captures.values():
            self.assertTrue(any(row[0] == self.qids["singleton"] for row in capture["ranked"]))
            self.assertFalse(any(row["qid"] == self.qids["singleton"] for row in capture["stages"]))

    def test_changing_labels_does_not_change_frequencies_keys_or_ranked_pairs(self):
        changed = {rid: list(targets) for rid, targets in self.truth.items()}
        changed[self.qids["df"]] = ["S2-singleton-distractor", "S3-singleton-distractor"]
        changed[self.qids["singleton"]] = ["S2-df-0001", "S3-df-0001"]
        data = self._write_fixture("changed-labels", changed)
        _, captures = self._run_capture(data, self.root / "changed-labels-output")
        for key, baseline in self.captures.items():
            for field in ("frequencies", "selected", "ranked"):
                with self.subTest(partition=key, field=field):
                    self.assertEqual(baseline[field], captures[key][field])

    def test_source_with_zero_true_links_is_a_valid_slice(self):
        changed = {rid: [tid for tid in targets if tid.startswith("S2-")] for rid, targets in self.truth.items()}
        data = self._write_fixture("zero-source3-truth", changed)
        report, _ = self._run_capture(data, self.root / "zero-source3-output")
        for metric in report["metrics"]:
            if metric["source"] == "source3":
                self.assertEqual(metric["true_links"], 0)
                self.assertIsNone(metric["candidate_link_recall"])
                self.assertEqual(metric["known_nonmatch_candidates"], metric["candidate_pairs"])

    def test_unknown_ground_truth_target_stops_instead_of_injecting_a_candidate(self):
        changed = {rid: list(targets) for rid, targets in self.truth.items()}
        changed[self.qids["df"]] = ["S2-nonexistent-synthetic-target"]
        data = self._write_fixture("unknown-target", changed)
        with self.assertRaisesRegex(RuntimeError, "positive targets absent"):
            self._run_capture(data, self.root / "unknown-target-output")

    def test_different_duckdb_version_cannot_silently_resample_validation(self):
        with patch.object(diagnostics.duckdb, "__version__", "1.5.5"):
            with self.assertRaisesRegex(RuntimeError, "original DuckDB 1.3.2"):
                diagnostics.run_diagnostics(self.data, self.root / "wrong-version-output")

    def _copy_checkpoint(self, name):
        output = self.root/name
        shutil.copytree(self.root/'baseline-output',output,
                        ignore=shutil.ignore_patterns('scratch.duckdb*','spill','queries.jsonl'))
        return output

    def _resume(self, output, **kwargs):
        with patch.object(diagnostics,'TARGET_BATCH_ROWS',32), redirect_stdout(io.StringIO()):
            return diagnostics.run_diagnostics(kwargs.pop('data',self.data),output,
                per_country=kwargs.pop('per_country',100),memory_mb=kwargs.pop('memory_mb',128),**kwargs)

    def test_batched_normalization_matches_whole_population_and_bounds_input(self):
        audit = diagnostics.Audit(self.root/'normalization-only',128)
        self.addCleanup(audit.close)
        audit.run("CREATE TABLE invented(entity_id VARCHAR,business_name VARCHAR,business_address VARCHAR,country VARCHAR)")
        rows = [
            ('S2-u0','Café Private Ltd.','01 Main Road Apt 2',COUNTRY),
            ('S2-u1','नमस्ते कंपनी','द्वार १२३',COUNTRY),
            ('S2-u2',None,None,COUNTRY),
            ('S2-u3','___','½ 18B / 2 Street',COUNTRY),
            ('S2-u4','Road Company LLC','4 Avenue, 00012',COUNTRY),
            ('S2-u5','e\u0301cole inc inc','1/2 Unit 03 Road',COUNTRY),
            ('S2-u6','株式会社','',COUNTRY),
        ]
        audit.db.executemany('INSERT INTO invented VALUES (?,?,?,?)',rows)
        audit.run(f"CREATE TABLE expected AS {diagnostics.population_sql('invented')}")
        original = diagnostics.population_sql
        observed = []

        def inspect(relation):
            if relation=='normalization_batch':
                observed.append(audit.run('SELECT count(*) FROM normalization_batch').fetchone()[0])
            return original(relation)

        with patch.object(diagnostics,'population_sql',side_effect=inspect):
            count = diagnostics.normalize_targets_batched(audit,'invented',batch_rows=3)
        actual = audit.run('SELECT * EXCLUDE(diagnostic_batch) FROM normalized_targets ORDER BY rid').fetchall()
        expected = audit.run('SELECT * FROM expected ORDER BY rid').fetchall()
        self.assertEqual(actual,expected)
        self.assertEqual(count,len(rows))
        self.assertEqual(observed,[0,3,3,1])
        self.assertEqual(audit.run('SELECT diagnostic_batch,count(*) FROM normalized_targets GROUP BY 1 ORDER BY 1').fetchall(),[(0,3),(1,3),(2,1)])

    def test_batched_normalization_accepts_empty_target_population(self):
        audit = diagnostics.Audit(self.root/'empty-normalization',128)
        self.addCleanup(audit.close)
        audit.run('CREATE TABLE invented(entity_id VARCHAR,business_name VARCHAR,business_address VARCHAR,country VARCHAR)')
        self.assertEqual(diagnostics.normalize_targets_batched(audit,'invented',batch_rows=3),0)
        self.assertEqual(audit.run('SELECT count(*) FROM normalized_targets').fetchone()[0],0)
        with self.assertRaisesRegex(ValueError,'positive integer'):
            diagnostics.normalize_targets_batched(audit,'invented',batch_rows=0)

    def test_completed_resume_skips_retrieval_and_preserves_artifacts(self):
        output = self._copy_checkpoint('resume-complete')
        before = {p.name:(diagnostics.sha256_file(p),p.stat().st_mtime_ns)
                  for p in output.iterdir() if p.suffix in ('.parquet','.csv') or p.name.startswith('metrics_')}
        with (patch.object(diagnostics,'normalize_targets_batched',side_effect=AssertionError('must not normalize completed targets')),
              patch.object(diagnostics,'run_policy',side_effect=AssertionError('must not rerun completed retrieval'))):
            report = self._resume(output)
        for field in ('sample','profiles','metrics','comparisons'):
            self.assertEqual(report[field],self.report[field])
        self.assertEqual(report['resumed_partitions'],[[COUNTRY,'source2'],[COUNTRY,'source3']])
        after = {name:(diagnostics.sha256_file(output/name),(output/name).stat().st_mtime_ns) for name in before}
        self.assertEqual(before,after)

    def test_legacy_progress_resumes_only_complete_partition_and_replaces_partial_files(self):
        output = self._copy_checkpoint('resume-partial-legacy')
        saved = json.loads((output/'progress.json').read_text(encoding='utf-8'))
        legacy = {key:[r for r in saved[key] if r['source']!='source3']
                  for key in ('profiles','metrics','comparisons')}
        diagnostics.atomic_json(output/'progress.json',legacy)
        complete_file = output/f"stages_{diagnostics.part_name(COUNTRY,'source2','legacy')}.parquet"
        before = (diagnostics.sha256_file(complete_file),complete_file.stat().st_mtime_ns)
        # Leftovers from the unfinished source are not trusted or parsed.
        unfinished = output/f"metrics_{diagnostics.part_name(COUNTRY,'source3','legacy')}.json"
        unfinished.write_text('interrupted non-JSON output',encoding='utf-8')
        partials = [output/(filename+'.partial') for filename in (
            f"stages_{diagnostics.part_name(COUNTRY,'source3','legacy')}.parquet",
            f"misses_{diagnostics.part_name(COUNTRY,'source3','legacy')}.csv")]
        for partial in partials:
            partial.write_bytes(b'interrupted COPY')
        with patch.object(diagnostics,'normalize_targets_batched',wraps=diagnostics.normalize_targets_batched) as normalize:
            report = self._resume(output)
        self.assertEqual(normalize.call_count,1)
        self.assertIn('train_source3.tsv',normalize.call_args.args[1])
        self.assertEqual(report['resumed_partitions'],[[COUNTRY,'source2']])
        self.assertEqual(report['metrics'],self.report['metrics'])
        self.assertEqual(report['profiles'],self.report['profiles'])
        self.assertEqual(before,(diagnostics.sha256_file(complete_file),complete_file.stat().st_mtime_ns))
        self.assertTrue(all(not p.exists() for p in partials))
        upgraded = json.loads((output/'progress.json').read_text(encoding='utf-8'))
        self.assertEqual(upgraded['checkpoint_version'],diagnostics.CHECKPOINT_VERSION)
        self.assertEqual(len(upgraded['completed_partitions']),2)
        self.assertIn(unfinished.name,upgraded['artifacts'])

    def test_resume_rejects_changed_input_configuration_and_version_without_overwriting_manifest(self):
        output = self._copy_checkpoint('resume-manifest-mismatch')
        manifest = (output/'input_manifest.json').read_bytes()
        for override in ({'per_country':101},{'memory_mb':256}):
            with self.subTest(override=override), self.assertRaisesRegex(RuntimeError,'Input manifest mismatch'):
                self._resume(output,**override)
        for attribute,value in (('CAPS',(40,80)),('PREPROCESS_VERSION','changed-preprocessing'),
                                ('DIAGNOSTICS_VERSION','changed-diagnostics')):
            with self.subTest(attribute=attribute), patch.object(diagnostics,attribute,value):
                with self.assertRaisesRegex(RuntimeError,'Input manifest mismatch'):
                    self._resume(output)
        changed = self._write_fixture('changed-target-for-resume')
        _write_tsv(changed/'train'/'train_source2.tsv',SOURCE_HEADER,
                   self.sources[2]+[('S2-extra-unlabeled','new distractor','',COUNTRY)])
        with self.assertRaisesRegex(RuntimeError,'Input manifest mismatch'):
            self._resume(output,data=changed)
        self.assertEqual(manifest,(output/'input_manifest.json').read_bytes())

    def test_resume_rejects_changed_ranking_even_without_version_bump(self):
        output = self._copy_checkpoint('resume-semantics-mismatch')
        original = diagnostics.ranked_sql()
        with patch.object(diagnostics,'ranked_sql',return_value=original+' /* changed semantics */'):
            with self.assertRaisesRegex(RuntimeError,'semantics mismatch'):
                self._resume(output)

    def test_resume_rejects_modified_frozen_query_contents(self):
        output = self._copy_checkpoint('resume-changed-sample')
        frozen = output/'frozen_queries.parquet'
        partial = output/'changed-sample.parquet'
        with duckdb.connect() as db:
            db.execute(f"""COPY (SELECT * REPLACE('changed name' AS business_name)
              FROM read_parquet({diagnostics.quote(frozen)})) TO {diagnostics.quote(partial)} (FORMAT parquet)""")
        partial.replace(frozen)
        with self.assertRaisesRegex(RuntimeError,'Frozen sample differs'):
            self._resume(output)

    def test_resume_rejects_missing_or_changed_completed_artifact(self):
        for action in ('missing','changed'):
            output = self._copy_checkpoint(f'resume-artifact-{action}')
            artifact = output/f"metrics_{diagnostics.part_name(COUNTRY,'source2','legacy')}.json"
            if action=='missing':
                artifact.unlink()
            else:
                artifact.write_text('[]',encoding='utf-8')
            expected = 'missing artifact' if action=='missing' else 'artifact hash mismatch'
            with self.subTest(action=action), self.assertRaisesRegex(RuntimeError,expected):
                self._resume(output)

    def test_resume_rejects_incomplete_completion_inventory(self):
        output = self._copy_checkpoint('resume-incomplete-inventory')
        path = output/'progress.json'
        saved = json.loads(path.read_text(encoding='utf-8'))
        saved['metrics'].pop()
        diagnostics.atomic_json(path,saved)
        with self.assertRaisesRegex(RuntimeError,'Invalid completed-partition records'):
            self._resume(output)

    def test_legacy_resume_reconciles_stage_files_instead_of_trusting_progress_only(self):
        output = self._copy_checkpoint('resume-legacy-corrupt-stage')
        path = output/'progress.json'
        saved = json.loads(path.read_text(encoding='utf-8'))
        diagnostics.atomic_json(path,{key:saved[key] for key in ('profiles','metrics','comparisons')})
        stage = output/f"stages_{diagnostics.part_name(COUNTRY,'source2','legacy')}.parquet"
        partial = output/'changed-stage.parquet'
        with duckdb.connect() as db:
            db.execute(f"COPY (SELECT * FROM read_parquet({diagnostics.quote(stage)}) WHERE false) TO {diagnostics.quote(partial)} (FORMAT parquet)")
        partial.replace(stage)
        with self.assertRaisesRegex(RuntimeError,'Completed stages disagree'):
            self._resume(output)

    def test_nonempty_output_without_manifest_is_never_adopted(self):
        output = self.root/'unidentified-output'
        output.mkdir()
        (output/'progress.json').write_text('{}',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'no input manifest'):
            self._resume(output)


if __name__ == "__main__":
    unittest.main()
