"""Hash/row verification works before editable installation as well."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.data.contracts import DataError
from fedmaint.data.integrity import verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        records = verify_manifest(args.root)
    except DataError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"verified_files": len(records), "files": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
