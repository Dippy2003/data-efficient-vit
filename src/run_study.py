"""Run a data-efficiency study across data fractions and random seeds."""

import argparse


DEFAULT_FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
DEFAULT_SEEDS = [42, 123, 456]
MODEL_NAMES = ["vit_scratch", "cnn", "vit_pretrained"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def validate_args(args) -> None:
    if not args.fractions or any(not 0 < value <= 1 for value in args.fractions):
        raise SystemExit("Every fraction must be in (0, 1].")
    if not args.seeds:
        raise SystemExit("At least one seed is required.")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise SystemExit("Epochs/batch size must be positive and patience non-negative.")


if __name__ == "__main__":
    validate_args(parse_args())
