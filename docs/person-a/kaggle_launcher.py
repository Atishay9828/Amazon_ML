"""Private Kaggle CPU launcher for Person A's reproducible retrieval runs.

Upload the A-owned Python sources as ``person-a-code.zip`` in a private Kaggle
dataset, attach it and the private official data dataset to a script kernel,
and set MODE below to smoke, dev, holdout, train, test, or a shard such as
``test-0-of-4`` before ``kaggle kernels push``.
All output stays in /kaggle/working until explicitly downloaded by the team.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MODE = "smoke"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def run(*args: str) -> None:
    print(json.dumps({"command": list(args), "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    shard = re.fullmatch(r"(train|test)-(\d+)-of-(\d+)", MODE)
    if MODE not in {"smoke", "dev", "holdout", "train", "test"} and not shard:
        raise ValueError(f"invalid mode: {MODE}")
    base_mode = shard.group(1) if shard else MODE
    data_files = list(INPUT.rglob("train_source1.tsv"))
    code_files = list(INPUT.rglob("candidates.py"))
    if len(data_files) != 1 or len(code_files) != 1:
        print("Kaggle input mounts:", list(INPUT.rglob("*"))[:60], flush=True)
        raise FileNotFoundError("attach both private data and code datasets")
    data = data_files[0].parent.parent
    source = code_files[0].parent.parent
    run(sys.executable, "-m", "pip", "install", "--quiet", "anyascii==0.3.3", "psutil==7.1.0")
    import anyascii  # noqa: F401
    import numpy
    import scipy
    import sklearn
    import psutil
    print(json.dumps({"stage": "environment", "mode": MODE, "python": sys.version,
                      "numpy": numpy.__version__, "scipy": scipy.__version__,
                      "sklearn": sklearn.__version__, "psutil": psutil.__version__,
                      "cpu_count": os.cpu_count(), "data": str(data),
                      "code": str(source)}), flush=True)
    if base_mode == "smoke":
        run(sys.executable, "-m", "unittest", "discover", "-s", str(source / "tests"), "-v")
    out = WORK / f"person-a-{MODE}-pairs.tsv"
    split = "test" if base_mode == "test" else "train"
    args = [sys.executable, str(source / "src" / "candidates.py"),
            "--data-root", str(data), "--split", split,
            "--out", str(out), "--work-dir", str(WORK / f"person-a-{MODE}.work"),
            "--workers", "2" if base_mode == "smoke" else "4",
            "--cap", "32" if base_mode == "smoke" else "64"]
    if base_mode == "smoke":
        args += ["--limit-source1", "1000", "--limit-targets", "1000000"]
    if base_mode in {"dev", "holdout"}:
        args += ["--sample-split", MODE]
    if shard:
        args += ["--shard-index", shard.group(2), "--shard-count", shard.group(3)]
    if base_mode in {"dev", "holdout", "train"}:
        args += ["--report", str(WORK / f"person-a-{MODE}-report.json")]
    run(*args)
    print(json.dumps({"stage": "artifact", "mode": MODE, "path": str(out),
                      "bytes": out.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
