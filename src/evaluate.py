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

import numpy as np
import torch


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
