"""
Evaluation module for the Data-Efficient ViT project.

Produces three types of output for each trained model:
  1. Overall accuracy on the test set
  2. Confusion matrix (which classes get confused with which)
  3. Per-class precision, recall, F1

Having these separate from src/train.py means we can reload a saved
checkpoint and evaluate it without re-running training -- useful when
training took a long time (e.g. the pretrained ViT on the full dataset).
"""

import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple:
    """
    Run `model` over the full `loader` and collect every true label and
    predicted label into numpy arrays.

    This is the shared engine for all evaluation functions below: accuracy,
    confusion matrix, and per-class report all need the same (y_true, y_pred)
    pair -- computing it once here avoids passing the model over the test set
    multiple times.

    Parameters
    ----------
    model : nn.Module  (already on device, already in eval() mode)
    loader : DataLoader  (non-augmented transform -- see src/data.py)
    device : torch.device

    Returns
    -------
    (y_true, y_pred) : two 1-D numpy arrays of integer class indices,
    length = number of samples in the loader.
    """
    model.eval()
    all_true, all_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_true.append(labels.numpy())
        all_pred.append(preds)

    return np.concatenate(all_true), np.concatenate(all_pred)


def confusion_matrix_from_loader(model, loader, device, class_names=None) -> tuple:
    """
    Compute the confusion matrix for `model` on `loader`.

    A confusion matrix is an NxN grid where entry (i, j) = number of times
    a sample of true class i was predicted as class j. The diagonal is
    correct predictions; off-diagonal entries show where the model gets
    confused. For CIFAR-10 you'll typically see cat<->dog and
    automobile<->truck confusions, which are visually similar pairs -- a
    good observation to call out in your video.

    Parameters
    ----------
    model, loader, device : same as collect_predictions()
    class_names : list of str, optional
        e.g. CIFAR10_CLASSES from src/data.py. Returned as-is for use by
        the plotting code.

    Returns
    -------
    (cm, class_names) where cm is a 2-D numpy array of shape (N, N).
    """
    y_true, y_pred = collect_predictions(model, loader, device)
    cm = confusion_matrix(y_true, y_pred)
    return cm, class_names


def per_class_report(model, loader, device, class_names=None) -> str:
    """
    Return sklearn's classification_report: per-class precision, recall,
    F1, and support (number of test samples per class).

    Why all three metrics (not just accuracy): accuracy hides class
    imbalance effects and doesn't tell you *why* a model fails. A model
    that ignores every 'cat' image and guesses randomly still gets ~90%
    accuracy on the other 9 classes. Precision/recall/F1 break this down
    so you can say, e.g., "the from-scratch ViT has near-zero recall on
    cat and dog, confirming those are the visually hardest classes without
    enough training data."

    Parameters
    ----------
    model, loader, device : same as collect_predictions()
    class_names : list/tuple of str, optional

    Returns
    -------
    str : formatted table (ready to print or embed in a notebook cell)
    """
    y_true, y_pred = collect_predictions(model, loader, device)
    return classification_report(y_true, y_pred, target_names=class_names, digits=3)


def compute_accuracy(model, loader, device) -> float:
    """
    Return the top-1 accuracy (fraction correct) of `model` on `loader`.

    Top-1 means we take the class with the highest logit as the prediction
    and check whether it matches the true label -- the standard metric for
    single-label classification like CIFAR-10.

    Returns
    -------
    float in [0, 1]
    """
    y_true, y_pred = collect_predictions(model, loader, device)
    return float((y_true == y_pred).mean())


def compute_per_class_accuracy(model, loader, device, class_names=None) -> dict:
    """Return accuracy for each class, including classes with zero correct."""
    y_true, y_pred = collect_predictions(model, loader, device)
    labels = sorted(np.unique(y_true).tolist())
    names = class_names or [str(label) for label in labels]
    return {
        names[label]: float((y_pred[y_true == label] == label).mean())
        for label in labels
    }


@torch.no_grad()
def collect_confident_errors(model, loader, device, limit: int = 20) -> list:
    """Collect the most confident mistakes for qualitative error analysis."""
    model.eval()
    errors = []
    offset = 0
    for images, labels in loader:
        probabilities = torch.softmax(model(images.to(device)), dim=1).cpu()
        confidence, predictions = probabilities.max(dim=1)
        for index, (truth, pred, score) in enumerate(zip(labels, predictions, confidence)):
            if truth.item() != pred.item():
                errors.append({"index": offset + index, "true": truth.item(),
                               "predicted": pred.item(), "confidence": score.item()})
        offset += len(labels)
    return sorted(errors, key=lambda item: item["confidence"], reverse=True)[:limit]


def build_results_table(models_dict: dict, loader, device, class_names=None) -> list:
    """
    Evaluate every model in `models_dict` on `loader` and return a list of
    result rows -- one per model -- for the final comparison table.

    This is the last piece of the evaluation pipeline: the table it produces
    is the single summary that answers "which model won, and by how much?"
    for your report and explainer video.

    Parameters
    ----------
    models_dict : dict  {model_name: nn.Module}
        All 3 models, already loaded with their best checkpoints and moved
        to `device`. Use build_model() + load_checkpoint() to prepare each.
    loader : DataLoader
        Test loader (non-augmented).
    device : torch.device
    class_names : tuple/list of str, optional

    Returns
    -------
    list of dicts, each with keys:
        "model"     -- human-readable model name
        "accuracy"  -- float, top-1 accuracy on test set
        "macro_f1"  -- float, macro-averaged F1 (balances across classes)
    """
    from sklearn.metrics import f1_score

    rows = []
    for name, model in models_dict.items():
        y_true, y_pred = collect_predictions(model, loader, device)
        acc = float((y_true == y_pred).mean())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
        params = sum(p.numel() for p in model.parameters())
        rows.append({"model": name, "accuracy": acc, "macro_f1": macro_f1,
                     "parameters": params})
        print(f"[results] {name}: accuracy={acc:.4f} macro_f1={macro_f1:.4f}")

    return rows


def print_results_table(rows: list, save_path: str = "outputs/results_table.txt"):
    """
    Pretty-print the comparison table from build_results_table() and save
    it to a text file.

    Parameters
    ----------
    rows : list of dicts (output of build_results_table())
    save_path : str  path to save the table; directory created if needed.
    """
    header = f"{'Model':<20} {'Accuracy':>10} {'Macro F1':>10} {'Params':>12}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for row in rows:
        lines.append(
            f"{row['model']:<20} {row['accuracy']:>10.4f} {row['macro_f1']:>10.4f} "
            f"{row.get('parameters', 0):>12,}"
        )
    lines.append(sep)
    table_str = "\n".join(lines)

    print(table_str)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        f.write(table_str + "\n")
    print(f"Results table saved to {save_path}")
