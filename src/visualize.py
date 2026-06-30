"""
Visualization module for the Data-Efficient ViT project.

Produces three types of plots saved to outputs/figures/:
  1. Training curves  -- loss and accuracy per epoch for all 3 models
  2. Confusion matrix -- heatmap of per-class predictions
  3. Attention maps   -- pretrained ViT's attention overlaid on sample images

All functions save their outputs to disk so they can be embedded in the
notebook without re-running training. File names include the model name so
the 3 models' plots don't overwrite each other.
"""

import os

import matplotlib
matplotlib.use("Agg")   # no GUI window needed -- saves directly to file
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = "outputs/figures"

# Colours consistent across all plots so each model is always the same colour.
MODEL_COLORS = {
    "vit_scratch": "#e74c3c",    # red   -- underperformer
    "cnn": "#3498db",            # blue  -- baseline
    "vit_pretrained": "#2ecc71", # green -- best
}


def plot_training_curves(
    history_dir: str = "outputs/history",
    model_names: list = None,
    save_path: str = None,
) -> str:
    """
    Load saved JSON training histories and plot loss + accuracy curves for
    all 3 models on a 2-panel figure (left: loss, right: accuracy).

    Why this matters: the curves show not just *final* accuracy but *how*
    each model learns. The pretrained ViT typically starts much higher (it
    already knows useful features), while the from-scratch ViT and CNN both
    start near random-chance (10% for CIFAR-10) and climb slowly. Showing
    this trajectory is the most compelling part of the data-efficiency story.

    Parameters
    ----------
    history_dir : str
        Directory containing JSON files written by src/train.save_history().
    model_names : list of str, optional
        Models to include. Defaults to all 3. Skip models whose history
        files don't exist (e.g. if you haven't trained all 3 yet).
    save_path : str, optional
        Where to save the PNG. Defaults to outputs/figures/training_curves.png.

    Returns
    -------
    str : path the figure was saved to.
    """
    import json

    if model_names is None:
        model_names = ["vit_scratch", "cnn", "vit_pretrained"]
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "training_curves.png")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    for name in model_names:
        path = os.path.join(history_dir, f"{name}_history.json")
        if not os.path.exists(path):
            print(f"[visualize] skipping {name} -- history file not found at {path}")
            continue

        with open(path) as f:
            h = json.load(f)

        color = MODEL_COLORS.get(name, None)
        epochs = range(1, len(h["train_loss"]) + 1)

        ax_loss.plot(epochs, h["train_loss"], color=color, linestyle="-",  label=f"{name} train")
        ax_loss.plot(epochs, h["val_loss"],   color=color, linestyle="--", label=f"{name} val")
        ax_acc.plot( epochs, h["train_acc"],  color=color, linestyle="-",  label=f"{name} train")
        ax_acc.plot( epochs, h["val_acc"],    color=color, linestyle="--", label=f"{name} val")

    for ax, title, ylabel in [
        (ax_loss, "Loss per epoch",     "Cross-entropy loss"),
        (ax_acc,  "Accuracy per epoch", "Accuracy"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Training curves — solid=train, dashed=val", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved training curves to {save_path}")
    return save_path


def plot_confusion_matrix(cm, class_names, model_name: str, save_path: str = None) -> str:
    """
    Plot a confusion matrix as a colour heatmap and save it to PNG.

    Each cell (i, j) shows how many test samples of true class i were
    predicted as class j. The diagonal = correct predictions (lighter =
    more), off-diagonal = errors. Normalising by row (dividing each row by
    its sum) converts raw counts to fractions, which is easier to read when
    class sizes differ.

    Parameters
    ----------
    cm : 2-D numpy array  (output of confusion_matrix_from_loader())
    class_names : list/tuple of str
    model_name : str  used in the figure title and filename
    save_path : str, optional  defaults to outputs/figures/<model_name>_confusion.png

    Returns
    -------
    str : path the figure was saved to.
    """
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, f"{model_name}_confusion.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Normalise: row sums → 1.0 so the colour scale is always [0, 1]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    n = len(class_names) if class_names else cm.shape[0]
    fig, ax = plt.subplots(figsize=(n * 0.9, n * 0.8))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if class_names:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)

    # Annotate each cell with its normalised fraction
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved confusion matrix to {save_path}")
    return save_path
