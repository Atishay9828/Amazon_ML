"""Regenerate the one-code-cell notebook after editing one_cell_solution.py."""

from pathlib import Path

import nbformat


HERE = Path(__file__).resolve().parent
intro = """# AJ's dual T4 entity-resolution experiment

Attach the official `student_resource` archive as a **private Kaggle Dataset**, select **GPU T4 x2**, and run the notebook. The single code cell below discovers the attached `dataset/` directory, generates bounded candidates for all test Source 1 records, fits separate Source 2 and Source 3 XGBoost models concurrently on `cuda:0` and `cuda:1`, tunes macro F0.5 on held-out training Source 1 records, and writes both required TSVs under `/kaggle/working/aj_entity_resolution/output/`.

The candidate-generation step runs on CPU/DuckDB and may take substantial time at this dataset's ~24 million-record scale. Inspect measured pair counts, candidate recall, and validation score before uploading anything. The official data, generated TSVs, and saved model files must stay private. Local structural checks are not a full Kaggle run or a leaderboard score.
"""
nb = nbformat.v4.new_notebook(
    cells=[
        nbformat.v4.new_markdown_cell(intro),
        nbformat.v4.new_code_cell((HERE / "one_cell_solution.py").read_text(encoding="utf-8")),
    ],
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbformat.validate(nb)
(HERE / "aj_dual_t4_entity_resolution.ipynb").write_text(nbformat.writes(nb), encoding="utf-8")
