"""Validate both datasets, prepare samples/preprocessors, then publish metadata."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.data.contracts import PipelineConfig
from fedmaint.data.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase01.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase01")
    args = parser.parse_args()
    catalog = prepare(ROOT, PipelineConfig.load(args.config), args.output)
    print(json.dumps({"verified_files": sum(len(d["files"]) for d in catalog["datasets"]),
                      "config_sha256": catalog["config_sha256"],
                      "factories": [{"id": f["factory_id"], "role": f["role"], "units": f["units"],
                                     "samples": f["samples"]} for f in catalog["factories"]],
                      "metadata": str(args.output / "metadata.json")}, indent=2))


if __name__ == "__main__":
    main()
