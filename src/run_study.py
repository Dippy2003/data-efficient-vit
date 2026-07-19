"""Run a data-efficiency study across data fractions and random seeds."""

import argparse
import csv
import json
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path


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
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse completed checkpoints when resuming a study")
    return parser.parse_args()


def validate_args(args) -> None:
    if not args.fractions or any(not 0 < value <= 1 for value in args.fractions):
        raise SystemExit("Every fraction must be in (0, 1].")
    if not args.seeds:
        raise SystemExit("At least one seed is required.")
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise SystemExit("Epochs/batch size must be positive and patience non-negative.")


def run_one(fraction: float, seed: int, args, device) -> list:
    """Train and evaluate all requested models for one study condition."""
    from src.data import get_dataloaders
    from src.evaluate import build_results_table
    from src.models import build_model
    from src.train import load_checkpoint, set_seed, train_model

    set_seed(seed)
    loaders = get_dataloaders(subset_fraction=fraction, batch_size=args.batch_size,
                              num_workers=0, seed=seed)
    condition = f"fraction_{fraction:g}/seed_{seed}"
    root = Path("outputs/studies") / condition
    results = []
    for name in args.models:
        model = build_model(name).to(device)
        checkpoint_dir = root / "checkpoints"
        checkpoint = checkpoint_dir / f"{name}_best.pth"
        if not (args.skip_existing and checkpoint.exists()):
            train_model(name, model, loaders, device, args.epochs, args.patience,
                        str(checkpoint_dir), str(root / "history"))
        model = load_checkpoint(model, name, device, str(root / "checkpoints"))
        row = build_results_table({name: model}, loaders["test"], device)[0]
        results.append({**row, "fraction": fraction, "seed": seed})
    return results


def aggregate_results(rows: list) -> list:
    """Calculate mean and sample standard deviation across random seeds."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["fraction"])].append(row)
    summary = []
    for (model, fraction), values in sorted(groups.items()):
        accuracies = [item["accuracy"] for item in values]
        f1_scores = [item["macro_f1"] for item in values]
        summary.append({
            "model": model, "fraction": fraction, "runs": len(values),
            "accuracy_mean": mean(accuracies),
            "accuracy_std": stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "macro_f1_mean": mean(f1_scores),
            "macro_f1_std": stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
        })
    return summary


def main() -> None:
    from src.train import get_device

    args = parse_args()
    validate_args(args)
    device = get_device()
    rows = []
    for fraction in args.fractions:
        for seed in args.seeds:
            rows.extend(run_one(fraction, seed, args, device))
    output = Path("outputs/studies")
    output.mkdir(parents=True, exist_ok=True)
    (output / "study_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = aggregate_results(rows)
    (output / "study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from src.visualize import plot_data_efficiency
    plot_data_efficiency(summary)
    with (output / "study_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Completed {len(rows)} model runs.")


if __name__ == "__main__":
    main()
