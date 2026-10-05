"""Build local map/analytics data without retraining the existing estimator."""

import argparse
from pathlib import Path

from .config import ARTIFACTS_PATH, DATA_PATH, EXPLORER_PATH, MODEL_PATH
from .explorer import prepare_explorer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=EXPLORER_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--evaluation", type=Path, default=ARTIFACTS_PATH / "evaluation.json")
    parser.add_argument("--predictions", type=Path, default=ARTIFACTS_PATH / "test_predictions.csv")
    args = parser.parse_args()
    try:
        artifact = prepare_explorer(args.data, args.output, args.model, args.evaluation, args.predictions)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Explorer preparation failed: {exc}\n")
    print(f"Prepared {len(artifact['listings']):,} in-scope listings; coordinates rounded to 0.01 degrees.")
    print("Verified model evaluation included." if artifact["evaluation"]["available"] else artifact["evaluation"]["reason"])


if __name__ == "__main__":
    main()
