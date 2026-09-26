"""Stage an A-only source ZIP for a *private* Kaggle code dataset.

Run from anywhere: python docs/person-a/package_kaggle_code.py --out D:/temp/amazon-ml-code-upload
The staged folder contains only Python source/tests, pipeline configs, and Kaggle metadata.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "code" / "business_entity_resolution"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "person-a-code.zip"
    files = sorted([*(PACKAGE / "src").glob("*.py"), *(PACKAGE / "tests").glob("*.py"),
                    *(PACKAGE / "configs").glob("*.json")])
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    metadata = {
        "title": "Amazon ML Person A private code",
        "id": "mridulnegi2005/amazon-ml-person-a-code",
        "licenses": [{"name": "Apache 2.0"}],
        "subtitle": "Private candidate generation code for the Amazon ML challenge",
        "description": "Private team code artifact. No external business data or labels.",
    }
    (args.out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"zip": str(destination), "files": [str(p.relative_to(ROOT)) for p in files],
                      "bytes": destination.stat().st_size}))


if __name__ == "__main__":
    main()
