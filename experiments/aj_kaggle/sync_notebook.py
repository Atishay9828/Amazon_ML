"""Build self-contained notebooks with exactly one code cell each."""

from pathlib import Path

import nbformat


HERE = Path(__file__).resolve().parent
HELPERS = (HERE / "output_assembly.py").read_text(encoding="utf-8")
SOURCE = (HERE / "one_cell_solution.py").read_text(encoding="utf-8")
IMPORT = "from output_assembly import assemble_outputs, ao_atomic_json, ao_fingerprint, ao_sha256\n"
if SOURCE.count(IMPORT) != 1:
    raise RuntimeError("Expected exactly one assembly import to inline in the full notebook")
FULL_CODE = HELPERS + "\n\n" + SOURCE.replace(IMPORT, "")


def write_notebook(filename, intro, code):
    # Check Python syntax without importing packages or executing any pipeline.
    compile(code, filename, "exec")
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(code)],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbformat.validate(nb)
    (HERE / filename).write_text(nbformat.writes(nb), encoding="utf-8")


write_notebook("aj_dual_t4_entity_resolution.ipynb", """# AJ's dual T4 entity-resolution experiment

Attach the official `student_resource` archive as a **private Kaggle Dataset**, select **GPU T4 x2**, and run the notebook. This single code cell discovers the attached `dataset/` directory, generates bounded candidates for all test Source 1 records, fits separate Source 2 and Source 3 XGBoost models concurrently on `cuda:0` and `cuda:1`, tunes macro F0.5 on held-out training Source 1 records, and writes both required TSVs under `/kaggle/working/aj_entity_resolution/output/`.

Candidate generation and output assembly use CPU/DuckDB. Output assembly sorts only IDs, with one thread and a 768 MB DuckDB budget, then streams one Source 1 row at a time. This budget is not a hard cap on total process memory. Completed inference partitions receive a completion record. The final output record is `output_completion.json`; `run_manifest.json` also records metrics and package versions.

**If your previous run finished inference and died during export, use `aj_recover_outputs.ipynb` instead. Do not rerun this full cell to recover files.**

Inspect measured candidate recall, validation score, runtime, disk use, and output validation before uploading. Challenge data, predictions, and models stay private. AJ's September 26 log confirms output recovery from the saved legacy Parquets. This revised full notebook, including the new inference checkpoints, still needs a fresh complete Kaggle run; no leaderboard score is claimed. Both TSVs receive streaming checks, including target ID existence. The separate organizer validator is not run automatically because it materializes large Python collections.
""", FULL_CODE)

RECOVERY_RUNNER = '''
# Settings for AJ's completed September 26 run. No training or inference occurs.
# If the files are restored elsewhere, set these two directories explicitly.
RECOVERY_DATA_ROOT = None  # Auto-discover the one official dataset under /kaggle/input.
RECOVERY_WORK_ROOT = "/kaggle/working/aj_entity_resolution"

# Counts copied from AJ's completed inference log, NOT estimated model outputs.
# The old writer could close a partial file with a valid footer. Matching these
# counts is required when that run has no per-partition completion sidecars.
EXPECTED_ACCEPTED_COUNTS = {
    "source2_e3772ac4b4_v3_64_512_3_40.parquet": 418_300,
    "source3_e3772ac4b4_v3_64_512_3_40.parquet": 471_367,
    "source2_967ce367d8_v3_64_512_3_40.parquet": 1_044_009,
    "source3_967ce367d8_v3_64_512_3_40.parquet": 1_056_068,
    "source2_aa30935544_v3_64_512_3_40.parquet": 951_167,
    "source3_aa30935544_v3_64_512_3_40.parquet": 1_012_640,
}
recovery_result = recover_saved_outputs(
    data_root=RECOVERY_DATA_ROOT,
    work_root=RECOVERY_WORK_ROOT,
    candidate_tag="v3_64_512_3_40",
    expected_accepted_counts=EXPECTED_ACCEPTED_COUNTS,
)
print(json.dumps(recovery_result, indent=2))
'''
RECOVERY_CODE = HELPERS + "\n\n" + RECOVERY_RUNNER
(HERE / "recover_outputs.py").write_text(RECOVERY_CODE, encoding="utf-8")
write_notebook("aj_recover_outputs.ipynb", """# Recover AJ's completed inference run

**Run only this code cell in the existing Kaggle notebook session that still has `/kaggle/working/aj_entity_resolution/`.** Copy the full code cell below (or all of `recover_outputs.py`) into a new cell there. Opening this notebook in a separate Kaggle session does not transfer the old session's saved files. If you must start another session, first preserve and restore the entire run directory and attach the same official dataset.

Restart the Python kernel if needed, but avoid resetting/deleting the Kaggle session or its files. Do not rerun the original training cell. This recovery cell imports no GPU framework and performs no training or inference; the two T4 models have already done that work.

The cell requires all six saved candidate Parquets in `test_candidates/`, all six accepted Parquets in `accepted/`, and the official test TSVs. Legacy accepted counts are checked against your supplied completion log (4,953,551 total accepted links). The old files have no completion sidecars, so this is log-based recovery evidence, not a reconstruction of missing model provenance. Missing files or count mismatches stop recovery.

This sorts narrow ID files with a 768 MB DuckDB budget and streams both TSVs without a global string aggregation. It needs additional disk space for sorted IDs, spill files, and TSVs; progress includes RAM and free disk. It checks exact Source 1 coverage, unique IDs, target existence, and matches being a subset of candidates. The organizer validator is not run automatically.

After **OUTPUT COMPLETE**, download `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and `output_completion.json` from the run directory. Only `matching_results.tsv` goes to the live leaderboard. Preserve the rest of the run for the final reproducible package. AJ's supplied September 26 log reports recovery completed at 04:41:11 with 1,732,544 Source 1 rows and 4,953,551 accepted links; the assembler checks passed. The separate organizer validator did not run, and no leaderboard score has been supplied. If this is that same completed run, download its files without rerunning this cell.
""", RECOVERY_CODE)
