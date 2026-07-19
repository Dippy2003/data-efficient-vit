"""
End-to-end experiment runner with a command-line interface.

Trains, evaluates, and visualises all three models in one command.
Results are saved to outputs/ automatically.

Usage
-----
  # Fast dev run (5% data, 2 epochs) -- default
  python -m src.run_experiment

  # Full graded run (100% data, 15 epochs)
  python -m src.run_experiment --mode final

  # Custom: train only 2 specific models for 5 epochs on 10% of the data
  python -m src.run_experiment --models vit_scratch cnn --epochs 5 --subset 0.1

  # Skip training, just re-evaluate existing checkpoints
  python -m src.run_experiment --skip-train

  # Skip training AND visualisation, just print the results table
  python -m src.run_experiment --skip-train --skip-viz

Arguments
---------
  --mode        dev | final  (shorthand presets)
  --models      one or more of: vit_scratch  cnn  vit_pretrained
  --epochs      number of training epochs (overrides --mode)
  --subset      fraction of training data to use, 0 < x <= 1.0
  --batch-size  dataloader batch size (default: 64)
  --skip-train  load existing checkpoints, skip training entirely
  --skip-viz    skip generating visualisation plots
"""

import argparse
import os
import sys
import time


ALL_MODELS = ["vit_scratch", "cnn", "vit_pretrained"]


def parse_args():
    parser = argparse.ArgumentParser(
        prog="python -m src.run_experiment",
        description="Train, evaluate, and visualise all three models on CIFAR-10.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "final"],
        default="dev",
        help="Preset: dev=5%% data/2 epochs, final=100%% data/15 epochs (default: dev)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=None,
        metavar="MODEL",
        help="Models to run. Default: all three. Choices: vit_scratch cnn vit_pretrained",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides --mode preset)",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Fraction of training data to use, e.g. 0.1 for 10%% (overrides --mode preset)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        dest="batch_size",
        help="Dataloader batch size (default: 64)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stopping patience; 0 disables it")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; load existing checkpoints from outputs/checkpoints/",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualisation plots (results table is always printed)",
    )
    return parser.parse_args()


def resolve_config(args):
    """Merge --mode preset with any explicit overrides."""
    presets = {
        "dev":   {"subset_fraction": 0.05, "num_epochs": 2},
        "final": {"subset_fraction": 1.0,  "num_epochs": 15},
    }
    cfg = presets[args.mode].copy()
    if args.epochs is not None:
        if args.epochs < 1:
            parser_error("--epochs must be at least 1")
        cfg["num_epochs"] = args.epochs
    if args.subset is not None:
        if not (0 < args.subset <= 1.0):
            parser_error(f"--subset must be between 0 and 1.0, got {args.subset}")
        cfg["subset_fraction"] = args.subset
    cfg["models"]     = args.models or ALL_MODELS
    cfg["batch_size"] = args.batch_size
    if cfg["batch_size"] < 1:
        parser_error("--batch-size must be at least 1")
    cfg["seed"]       = args.seed
    cfg["patience"]   = args.patience
    if cfg["patience"] < 0:
        parser_error("--patience cannot be negative")
    cfg["skip_train"] = args.skip_train
    cfg["skip_viz"]   = args.skip_viz
    return cfg


def parser_error(message: str) -> None:
    """Print a consistent command-line validation error and exit."""
    print(f"[error] {message}")
    raise SystemExit(2)


def print_config(cfg, device):
    """Print a summary of the run configuration before starting."""
    bar = "=" * 55
    print(f"\n{bar}")
    print("  Data-Efficient ViT Experiment")
    print(bar)
    print(f"  Device          : {device}")
    print(f"  Models          : {', '.join(cfg['models'])}")
    print(f"  Subset fraction : {cfg['subset_fraction']:.0%} of CIFAR-10 training set")
    print(f"  Epochs          : {cfg['num_epochs']}")
    print(f"  Batch size      : {cfg['batch_size']}")
    print(f"  Random seed     : {cfg['seed']}")
    print(f"  Early stopping  : {cfg['patience'] or 'disabled'}")
    print(f"  Skip training   : {cfg['skip_train']}")
    print(f"  Skip viz        : {cfg['skip_viz']}")
    print(f"{bar}\n")


def run_training(cfg, loaders, device):
    """Train each requested model and return a dict of trained models."""
    from src.models import build_model
    from src.train  import load_checkpoint, train_model

    trained = {}
    for name in cfg["models"]:
        print(f"\n{'='*55}")
        print(f"  Training: {name}")
        print(f"{'='*55}")
        t0 = time.time()
        model = build_model(name, num_classes=10, img_size=224).to(device)
        train_model(name, model, loaders, device, num_epochs=cfg["num_epochs"],
                    patience=cfg["patience"])
        model = load_checkpoint(model, name, device)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f} min")
        trained[name] = model
    return trained


def load_checkpoints(cfg, device):
    """Load saved checkpoints for the requested models."""
    from src.models import build_model
    from src.train  import load_checkpoint

    loaded = {}
    missing = []
    for name in cfg["models"]:
        ckpt = os.path.join("outputs", "checkpoints", f"{name}_best.pth")
        if not os.path.exists(ckpt):
            missing.append(name)
            continue
        m = build_model(name, num_classes=10, img_size=224).to(device)
        loaded[name] = load_checkpoint(m, name, device)
    if missing:
        print(f"\n[warning] No checkpoint found for: {', '.join(missing)}")
        print("  Run without --skip-train first to generate them.\n")
    return loaded


def run_evaluation(models, loaders, device):
    """Print results table and per-class reports."""
    from src.data    import CIFAR10_CLASSES
    from src.evaluate import (build_results_table, print_results_table,
                               per_class_report)

    print(f"\n{'='*55}")
    print("  Evaluation — Results Table")
    print(f"{'='*55}")
    rows = build_results_table(models, loaders["test"], device, CIFAR10_CLASSES)
    print_results_table(rows)

    print(f"\n{'='*55}")
    print("  Per-class Precision / Recall / F1")
    print(f"{'='*55}")
    for name, model in models.items():
        print(f"\n-- {name} --")
        print(per_class_report(model, loaders["test"], device, CIFAR10_CLASSES))
    return rows


def run_visualizations(models, loaders, device):
    """Generate and save all plots."""
    from src.data     import CIFAR10_CLASSES
    from src.evaluate import confusion_matrix_from_loader
    from src.visualize import (plot_training_curves, plot_confusion_matrix,
                                plot_sample_predictions, plot_attention_overlay)

    print(f"\n{'='*55}")
    print("  Visualizations")
    print(f"{'='*55}")

    # Training curves (reads JSON histories saved during training)
    path = plot_training_curves(
        model_names=list(models.keys()),
        save_path="outputs/figures/training_curves.png",
    )
    print(f"  Saved: {path}")

    for name, model in models.items():
        # Confusion matrix
        cm, classes = confusion_matrix_from_loader(
            model, loaders["test"], device, CIFAR10_CLASSES
        )
        path = plot_confusion_matrix(
            cm, classes, name,
            save_path=f"outputs/figures/{name}_confusion.png",
        )
        print(f"  Saved: {path}")

        # Sample predictions grid
        path = plot_sample_predictions(
            model, loaders["test"], device, CIFAR10_CLASSES, name,
            save_path=f"outputs/figures/{name}_predictions.png",
        )
        print(f"  Saved: {path}")

        # Attention maps for ViT models only
        if "vit" in name:
            images, labels = next(iter(loaders["test"]))
            sample_img   = images[:1].to(device)
            sample_class = CIFAR10_CLASSES[labels[0].item()]
            path = plot_attention_overlay(
                model, sample_img,
                class_name=sample_class,
                model_name=name,
                save_path=f"outputs/figures/{name}_attention.png",
            )
            print(f"  Saved: {path}")


def main():
    args   = parse_args()
    cfg    = resolve_config(args)

    from src.data  import get_dataloaders
    from src.train import get_device, set_seed

    set_seed(cfg["seed"])
    device = get_device()
    print_config(cfg, device)

    print("[1/3] Loading data...")
    loaders = get_dataloaders(
        dataset_name    = "cifar10",
        data_root       = "data",
        image_size      = 224,
        batch_size      = cfg["batch_size"],
        subset_fraction = cfg["subset_fraction"],
        seed            = cfg["seed"],
        num_workers     = 0,
    )
    print(f"      Train: {len(loaders['train'].dataset)}  "
          f"Val: {len(loaders['val'].dataset)}  "
          f"Test: {len(loaders['test'].dataset)}")

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n[2/3] Training..." if not cfg["skip_train"] else "\n[2/3] Loading checkpoints...")
    if cfg["skip_train"]:
        models = load_checkpoints(cfg, device)
    else:
        models = run_training(cfg, loaders, device)

    if not models:
        print("[error] No models available. Exiting.")
        sys.exit(1)

    # ── Evaluation ────────────────────────────────────────────────────────────
    rows = run_evaluation(models, loaders, device)

    from src.experiment import (append_results_csv, environment_info,
                                make_run_id, save_run_record)
    run_id = make_run_id(cfg)
    record_path = save_run_record({
        "run_id": run_id,
        "config": cfg,
        "environment": environment_info(),
        "results": rows,
    })
    print(f"Experiment record saved to {record_path}")
    append_results_csv(run_id, cfg, rows)

    # ── Visualizations ────────────────────────────────────────────────────────
    if not cfg["skip_viz"]:
        print("\n[3/3] Generating visualizations...")
        run_visualizations(models, loaders, device)
    else:
        print("\n[3/3] Skipping visualizations (--skip-viz).")

    print("\nDone. All outputs saved to outputs/")


if __name__ == "__main__":
    main()
