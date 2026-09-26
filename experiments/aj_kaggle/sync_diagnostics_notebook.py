"""Generate AJ's staged preprocessing notebook and standalone equivalent."""

import hashlib
from pathlib import Path

import nbformat


HERE=Path(__file__).resolve().parent
prep=(HERE/'preprocessing.py').read_text(encoding='utf-8')
audit=(HERE/'diagnostics.py').read_text(encoding='utf-8')
presentation=(HERE/'diagnostic_reporting.py').read_text(encoding='utf-8')
tests=(HERE/'tests/test_preprocessing.py').read_text(encoding='utf-8')
bootstrap=(HERE/'statistical_diagnostics.py').read_text(encoding='utf-8')


def remove_once(text, fragment):
    if text.count(fragment)!=1:
        raise RuntimeError(f'Expected one notebook-only removal: {fragment[:80]}')
    return text.replace(fragment,'')


audit=remove_once(audit,'from preprocessing import PREPROCESS_VERSION, normalized_sql\n')
audit=audit.split("\nif __name__ == '__main__':\n")[0]
presentation=remove_once(presentation,'from diagnostics import Audit, atomic_json, part_name, population_sql, profile, quote, read_tsv\n')
presentation=presentation.split("\nif __name__=='__main__':\n")[0]
tests=remove_once(tests,'from __future__ import annotations\n')
tests=remove_once(tests,'from experiments.aj_kaggle.preprocessing import PREPROCESS_VERSION, normalized_sql\n')
tests=tests.split('\nif __name__ == "__main__":\n')[0]
bootstrap=remove_once(bootstrap,'from diagnostics import CAPS, part_name, quote\n')
digest=hashlib.sha256((prep+audit+presentation+tests+bootstrap).encode()).hexdigest()
runner='''
# ------------------------- EDITABLE SETTINGS -------------------------
DATA_ROOT = locate_data()
DIAGNOSTIC_OUTPUT = (Path('/kaggle/working/aj_preprocessing_ablations')
                     if Path('/kaggle/working').exists()
                     else Path.cwd()/'work'/'aj_preprocessing_ablations')
QUERIES_PER_COUNTRY = 1000
MEMORY_MB = 512
REUSE_COMPLETED_REPORT = True
# ---------------------------------------------------------------------

suite=unittest.defaultTestLoader.loadTestsFromTestCase(PreprocessingTests)
test_result=unittest.TextTestRunner(verbosity=1).run(suite)
if not test_result.wasSuccessful():
    raise RuntimeError('Preprocessing tests failed; diagnostic stopped')

report_path=DIAGNOSTIC_OUTPUT/'report.json'
report=None
if REUSE_COMPLETED_REPORT and report_path.is_file():
    saved=json.loads(report_path.read_text(encoding='utf-8'))
    manifest=saved.get('manifest',{})
    compatible=(manifest.get('diagnostics_version')==DIAGNOSTICS_VERSION
                and manifest.get('preprocessing_version')==PREPROCESS_VERSION
                and manifest.get('duckdb')==duckdb.__version__
                and manifest.get('per_country')==QUERIES_PER_COUNTRY
                and manifest.get('memory_mb')==MEMORY_MB
                and manifest.get('caps')==list(CAPS)
                and manifest.get('policies')==list(POLICIES)
                and saved.get('semantics_sha256')==semantics_sha256())
    if compatible:
        for source in manifest['sources']:
            path=DATA_ROOT/'train'/source['file']
            if not path.is_file() or sha256_file(path)!=source['sha256']:
                compatible=False
                break
    if compatible:
        report=saved
        print('Reusing completed diagnostic with matching input hashes and configuration.')
if report is None:
    report=run_diagnostics(DATA_ROOT,DIAGNOSTIC_OUTPUT,QUERIES_PER_COUNTRY,MEMORY_MB)

test_profile=test_data_profile(DATA_ROOT,DIAGNOSTIC_OUTPUT,QUERIES_PER_COUNTRY,MEMORY_MB)
figures=render_report(report,DIAGNOSTIC_OUTPUT,show=True)
print('Tests passed:',test_result.testsRun)
print('Report:',report_path)
print('Unlabeled test profile:',DIAGNOSTIC_OUTPUT/'test_profile.json')
print('Private examples and query membership stay in the working folder.')
'''
code=prep+'\n\n'+audit+'\n\n'+presentation+'\n\n'+tests+'\n\n'+bootstrap+'\n\n'+runner
compile(code,'aj_preprocessing_diagnostics.ipynb','exec')
intro='''# AJ — preprocessing and retrieval diagnostics

## Goal

Measure why true links disappear before matching, and whether conservative preprocessing helps. This notebook uses **separate cells for setup, tests, retrieval diagnostics, unlabeled test profiling, and reporting**. It performs CPU text processing, tests, and candidate diagnostics; model training remains in the separate dual-T4 notebook.

## Data and method

- Attach the same existing `student_stuff` Kaggle dataset. Attaching that stored dataset does not upload it again. Raw data and any previous prediction files are preserved.
- Use DuckDB **1.3.2**, matching the completed Kaggle run, because the original development/validation split uses a version-dependent `hash()` function. This cell never installs or downgrades packages. If the version differs, use the documented isolated runtime or return to the matching Kaggle environment.
- Freeze up to **1,000 development Source 1 records per training country**, from original buckets 5–14. The original validation buckets 0–4 are excluded. Country sampling is deliberately balanced, so aggregate diagnostic recall is not a population estimate or the previous validation score.
- Scan **all training target records** in each source/country to count key frequencies, in batches of 20,000. Ground truth is joined only after candidates are selected and ranked.
- Compare original preprocessing, new preprocessing at DF<=64 / 3 token keys per channel, DF-only at 256 / 3 keys, key-budget-only at 64 / 6 keys, and the combined 256 / 6 setting. This controlled ablation isolates each retrieval knob. Measure recall and candidate workload at caps **40, 80, 160, and 500 per target source**.
- Attribute each lost positive to its first failed stage. Measure missing-address recall separately. A cap above the true match count and the presence of name-based keys do not guarantee retrieval.
- Combine Source 2 and Source 3 per Source 1 entity to report micro link recall, mean/median positive-entity recall, complete/partial/uncovered positive queries, singleton rate, and a singleton-aware oracle macro F0.5 ceiling. The oracle assumes a perfect classifier that removes every false candidate, so it is an optimistic retrieval ceiling rather than a model score.
- Profile test country counts and sampled normalization, including France. No France accuracy is inferred without labels.

## Checks and outputs

The tests cell runs 24 synthetic preprocessing checks. Candidate-policy contract checks and synthetic retrieval integration checks live in the repository. The retrieval cell checks development membership, ground-truth coverage, retrieval/diagnostic agreement, and reconciled loss counts. The notebook writes aggregate reports, figures, and private missed-link examples under `/kaggle/working/aj_preprocessing_ablations/` (or `work/aj_preprocessing_ablations/` locally).

Completed reports can be reused when input hashes, versions, and configuration match; the setting is visible at the start of the runner. Set `REUSE_COMPLETED_REPORT=False` for a fresh audit. A fresh run scans millions of target records and needs working disk space. The 512 MB DuckDB budget is not a hard cap on total process memory.

Run the cells from top to bottom the first time. After an interrupted retrieval run, keep the working directory and rerun setup, tests, and retrieval; compatible completed partitions can be reused. To redraw charts after a complete audit, run setup, settings/tests, and the final reporting cell. Test profiling is a separate scan and need not be repeated just to redraw charts.

The results are retrieval measurements. Model F0.5 and leaderboard changes require a later GPU training and validation experiment. Keep official records, private examples, saved models, and predictions out of the public repository.
'''
setup_and_tests, remaining = runner.split("report_path=DIAGNOSTIC_OUTPUT/'report.json'", 1)
retrieval, final_steps = remaining.split('test_profile=test_data_profile', 1)
retrieval = "report_path=DIAGNOSTIC_OUTPUT/'report.json'" + retrieval
test_profile_cell = "test_profile=test_data_profile" + final_steps.split('figures=render_report', 1)[0]
reporting_cell = """report_path=DIAGNOSTIC_OUTPUT/'report.json'
if not report_path.is_file():
    raise RuntimeError('Complete the retrieval cell before reporting.')
report=json.loads(report_path.read_text(encoding='utf-8'))
if 'entity_metrics' not in report:
    report['entity_metrics']=entity_level_metrics(report,DIAGNOSTIC_OUTPUT)
    atomic_json(report_path,report)
    atomic_json(DIAGNOSTIC_OUTPUT/'entity_metrics.json',report['entity_metrics'])
""" + 'figures=render_report' + final_steps.split('figures=render_report', 1)[1]
bootstrap_cell = '''\nreport_path=DIAGNOSTIC_OUTPUT/'report.json'
report=json.loads(report_path.read_text(encoding='utf-8'))
bootstrap_metrics=run_bootstrap_diagnostics(report,DIAGNOSTIC_OUTPUT,replicates=2000,seed=20260926)
bootstrap_path=DIAGNOSTIC_OUTPUT/'bootstrap_uncertainty.csv'
pd.DataFrame(bootstrap_metrics).to_csv(bootstrap_path,index=False)
cols=['scope','policy','cap','oracle_macro_f05','oracle_macro_f05_ci_low','oracle_macro_f05_ci_high',
      'micro_true_link_recall','micro_recall_ci_low','micro_recall_ci_high',
      'delta_oracle_f05_vs_legacy','delta_oracle_f05_ci_low','delta_oracle_f05_ci_high']
print(pd.DataFrame(bootstrap_metrics).query("scope == 'ALL_COUNTRIES_STRATIFIED'")[cols].to_string(index=False))

# Transfer only aggregate evidence; never include IDs, records, examples, or candidate rows.
import zipfile
from IPython.display import FileLink, display
bundle_readme=DIAGNOSTIC_OUTPUT/'aggregate_bundle_README.txt'
bundle_readme.write_text('\\n'.join([
  'Private aggregate diagnostic export.',
  'Balanced training-development diagnostic sample: up to 1,000 Source 1 queries per training country; all target-country records indexed.',
  'Bootstrap intervals resample Source 1 queries within country; they do not estimate leaderboard uncertainty.',
  'Oracle F0.5 is an optimistic upper bound assuming all false candidates are rejected.',
  'No raw records, sampled IDs, missed-link examples, stage Parquets, scratch database, candidate pairs, or predictions are included.'
]),encoding='utf-8')
bundle_path=DIAGNOSTIC_OUTPUT/'aggregate_diagnostics.zip'
bundle_files=['report.json','test_profile.json','summary_metrics.csv','metrics_by_country_source.csv',
  'entity_metrics.csv','paired_comparisons.csv','normalization_profile.csv','bootstrap_uncertainty.csv',
  'recall_vs_cap.png','loss_by_stage.png','oracle_f05_ceiling.png','aggregate_bundle_README.txt']
missing=[name for name in bundle_files if not (DIAGNOSTIC_OUTPUT/name).is_file()]
if missing:
    raise RuntimeError(f'Aggregate export is missing expected files: {missing}')
with zipfile.ZipFile(bundle_path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in bundle_files:
        archive.write(DIAGNOSTIC_OUTPUT/name,arcname=name)
print('Download aggregate-only evidence bundle:',bundle_path)
display(FileLink(str(bundle_path),result_html_prefix='Download aggregate-only diagnostic results: '))
'''
download_cell = '''
# Kaggle's FileLink may open a session-only proxy address in a new tab. This
# small aggregate-only data URI provides a direct browser download instead.
import base64
from IPython.display import HTML, display
if not bundle_path.is_file():
    raise FileNotFoundError(f'Run the aggregate export cell first: {bundle_path}')
payload=base64.b64encode(bundle_path.read_bytes()).decode('ascii')
display(HTML(f'<a download="aggregate_diagnostics.zip" href="data:application/zip;base64,{payload}">Download aggregate diagnostics to this computer</a>'))
'''
cells = [
    nbformat.v4.new_markdown_cell(intro),
    nbformat.v4.new_markdown_cell('## 1. Load the preprocessing and diagnostic functions\n\nThis cell defines the work; it does not scan the dataset or fit models.'),
    nbformat.v4.new_code_cell(prep+'\n\n'+audit+'\n\n'+presentation+'\n\n'+bootstrap),
    nbformat.v4.new_markdown_cell('## 2. Settings and preprocessing tests\n\nEdit the sample size and memory budget here. The existing Kaggle input is discovered automatically.'),
    nbformat.v4.new_code_cell(tests+'\n\n'+setup_and_tests),
    nbformat.v4.new_markdown_cell('## 3. Run or resume the training retrieval diagnostic\n\nThis is the expensive full-target scan. Preserve its working directory to reuse compatible completed work.'),
    nbformat.v4.new_code_cell(retrieval),
    nbformat.v4.new_markdown_cell('## 4. Profile the unlabeled test data\n\nA separate scan for country counts and normalization, including France. These are not accuracy estimates.'),
    nbformat.v4.new_code_cell(test_profile_cell),
    nbformat.v4.new_markdown_cell('## 5. Read the completed report and draw the figures\n\nThis cell reuses the saved report; it does not regenerate candidates.'),
    nbformat.v4.new_code_cell(reporting_cell),
    nbformat.v4.new_markdown_cell('## 6. Statistical robustness and aggregate export\n\nThis cell bootstraps Source 1 queries within country, reports paired uncertainty for retrieval metrics, and exports only aggregate tables and figures. It does not rerun candidate generation.'),
    nbformat.v4.new_code_cell(bootstrap_cell),
    nbformat.v4.new_markdown_cell('## 7. Download a local aggregate copy\n\nRun this after the export cell. The link contains only the aggregate ZIP, not raw records, sampled IDs, examples, candidates, or predictions.'),
    nbformat.v4.new_code_cell(download_cell),
]
for cell in cells:
    if cell.cell_type == 'code':
        compile(cell.source,'diagnostic_cell','exec')
notebook=nbformat.v4.new_notebook(cells=cells,
  metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
            'language_info':{'name':'python','version':'3.12'},'diagnostic_source_sha256':digest})
nbformat.validate(notebook)
(HERE/'aj_preprocessing_diagnostics.ipynb').write_text(nbformat.writes(notebook),encoding='utf-8')
(HERE/'preprocessing_diagnostics_cell.py').write_text(code,encoding='utf-8')
