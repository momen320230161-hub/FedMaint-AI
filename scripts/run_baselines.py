"""Execute a fixed Phase 02 plan: all train/validation selection, then test/held-out evaluation."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.models.contracts import TrainingConfig
from fedmaint.models.experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase02.json")
    parser.add_argument("--prepared", type=Path, default=ROOT / "outputs/phase01")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase02")
    args = parser.parse_args()
    result = run_experiments(ROOT, args.prepared, args.output, TrainingConfig.load(args.config))
    print(f"{result['status']}: {len(result['runs'])} runs in {result['runtime_seconds']:.2f}s; {args.output}")


if __name__ == "__main__":
    main()
