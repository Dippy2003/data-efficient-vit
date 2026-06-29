"""
Training loop for the Data-Efficient ViT project.

This module trains any of the 3 models from src/models.py using the
dataloaders from src/data.py. Kept model-agnostic on purpose: it just calls
model(images) and expects logits back, so the same training code works for
the CNN and both ViT variants without special-casing.
"""

import json
import os

import torch
import torch.nn as nn


class AverageMeter:
    """
    Tracks a running average of a value (loss or accuracy) across batches.

    Why this is needed: each batch gives us one loss number and one
    accuracy number, but we want the *epoch's* average, not just the last
    batch's. Weighting each update by `n` (batch size) keeps the average
    correct even if the last batch is smaller than the others (common when
    the dataset size isn't an exact multiple of batch_size).
    """

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def get_device() -> torch.device:
    """
    Pick the best available device and warn if we're stuck on CPU.

    Why this matters: training a ViT (even the "tiny" variant) on CPU is
    noticeably slower than a CNN of similar size, because self-attention
    does more matrix multiplication per layer than convolution. If you see
    this warning, either switch to a CUDA machine or expect epochs to take
    minutes instead of seconds -- use a small subset_fraction while developing.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[train] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[train] WARNING: No GPU found, training on CPU. This will be "
              "slow, especially for the ViT models. Use a small "
              "subset_fraction for development, or switch to a CUDA machine "
              "/ Google Colab GPU runtime for the final run.")
    return device


def train_one_epoch(model, loader, optimizer, device) -> dict:
    """
    Run one full pass over the training data: forward pass, loss, backward
    pass, optimizer step -- repeated for every batch in `loader`.

    Parameters
    ----------
    model : nn.Module
        Any of the 3 models from src/models.py. Must already be on `device`.
    loader : DataLoader
        Training DataLoader (the one with augmentation applied).
    optimizer : torch.optim.Optimizer
        Already constructed for this model's parameters.
    device : torch.device

    Returns
    -------
    dict with keys "loss" and "accuracy" -- the epoch's averages.
    """
    model.train()  # enables dropout / batchnorm update behaviour
    criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # accuracy: fraction of predictions matching the true label
        preds = logits.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item()

        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(batch_acc, n=images.size(0))

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Run the model over `loader` without updating weights -- used both for
    validation during training (to watch for overfitting) and for the
    final test-set evaluation.

    The @torch.no_grad() decorator disables gradient tracking, which is
    unnecessary here and would otherwise waste memory and compute.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
        Should use the *non*-augmented transform (see src/data.py) -- we
        want to measure real performance, not performance on randomly
        perturbed images.
    device : torch.device

    Returns
    -------
    dict with keys "loss" and "accuracy".
    """
    model.eval()  # disables dropout, freezes batchnorm running stats
    criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        batch_acc = (preds == labels).float().mean().item()

        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(batch_acc, n=images.size(0))

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


def get_optimizer(model_name: str, model: nn.Module) -> torch.optim.Optimizer:
    """
    Build the optimizer with sensible, model-appropriate defaults.

    Why this differs by model: transformers (both ViT variants) are
    typically trained with AdamW rather than plain SGD, since attention
    layers tend to train more stably with adaptive per-parameter learning
    rates. For the *pretrained* ViT we also use a smaller learning rate --
    fine-tuning should nudge the already-good pretrained weights gently,
    not overwrite them with large updates the way training from scratch
    requires.

    Parameters
    ----------
    model_name : str
        One of "vit_scratch", "cnn", "vit_pretrained" (matches
        src/models.py's MODEL_REGISTRY keys).
    model : nn.Module

    Returns
    -------
    torch.optim.Optimizer
    """
    if model_name == "vit_pretrained":
        lr = 1e-4  # small LR: fine-tune gently, don't destroy pretrained weights
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif model_name == "vit_scratch":
        lr = 1e-3  # larger LR: this model must learn everything from zero
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif model_name == "cnn":
        lr = 1e-3
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown model_name '{model_name}'.")


def save_history(history: dict, model_name: str, history_dir: str = "outputs/history") -> str:
    """
    Save a training history dict (train/val loss & accuracy per epoch) as
    JSON, so src/visualize.py can plot curves later without needing to
    retrain -- training the pretrained ViT especially can take a while, so
    we don't want to throw the numbers away after one notebook session.

    Parameters
    ----------
    history : dict
        Output of train_model(): keys "train_loss", "train_acc", "val_loss",
        "val_acc", each a list of floats (one per epoch).
    model_name : str
        Used to build the filename, e.g. "cnn" -> "outputs/history/cnn_history.json".
    history_dir : str
        Directory to save into; created if it doesn't exist.

    Returns
    -------
    str : path the history was saved to.
    """
    os.makedirs(history_dir, exist_ok=True)
    path = os.path.join(history_dir, f"{model_name}_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    return path


def save_checkpoint(model: nn.Module, model_name: str, checkpoint_dir: str = "outputs/checkpoints") -> str:
    """
    Save the model's weights to disk.

    Why we need this: training can take a while (especially the ViTs), and
    we don't want to lose the best version of a model just because a later
    epoch happened to overfit. Saving by `model_name` keeps the 3 models'
    checkpoints separate so they don't overwrite each other.

    Parameters
    ----------
    model : nn.Module
    model_name : str
        Used to build the filename, e.g. "cnn" -> "outputs/checkpoints/cnn_best.pth".
    checkpoint_dir : str
        Directory to save into; created if it doesn't exist.

    Returns
    -------
    str : path the checkpoint was saved to.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
    torch.save(model.state_dict(), path)
    return path


def train_model(model_name: str, model, loaders, device, num_epochs: int = 5) -> dict:
    """
    Full training loop: runs `num_epochs` epochs, evaluating on the
    validation set after each one, and prints progress.

    This is the function the notebook will actually call for each of the
    3 models. Returns a history dict so src/visualize.py can later plot
    loss/accuracy curves without re-running training.

    Parameters
    ----------
    model_name : str
        Used to pick the right optimizer settings (see get_optimizer).
    model : nn.Module
        Should already be moved to `device` by the caller.
    loaders : dict
        Output of get_dataloaders() -- needs "train" and "val" keys.
    device : torch.device
    num_epochs : int
        Keep this small (3-5) while developing; scale up for final results.

    Returns
    -------
    dict with keys "train_loss", "train_acc", "val_loss", "val_acc", each a
    list with one entry per epoch.
    """
    optimizer = get_optimizer(model_name, model)

    # Cosine decay smoothly lowers the learning rate to ~0 by the final
    # epoch. This is standard practice for both CNNs and ViTs: large steps
    # early help escape poor initial regions of the loss landscape, smaller
    # steps later help settle into a good minimum instead of overshooting it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_result = train_one_epoch(model, loaders["train"], optimizer, device)
        val_result = evaluate(model, loaders["val"], device)
        scheduler.step()

        history["train_loss"].append(train_result["loss"])
        history["train_acc"].append(train_result["accuracy"])
        history["val_loss"].append(val_result["loss"])
        history["val_acc"].append(val_result["accuracy"])

        print(
            f"[{model_name}] epoch {epoch}/{num_epochs} -- "
            f"train_loss={train_result['loss']:.4f} train_acc={train_result['accuracy']:.4f} "
            f"val_loss={val_result['loss']:.4f} val_acc={val_result['accuracy']:.4f}"
        )

        if val_result["accuracy"] > best_val_acc:
            best_val_acc = val_result["accuracy"]
            checkpoint_path = save_checkpoint(model, model_name)
            print(f"[{model_name}] new best val_acc={best_val_acc:.4f}, saved to {checkpoint_path}")

    history_path = save_history(history, model_name)
    print(f"[{model_name}] training history saved to {history_path}")

    return history


if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")

    meter = AverageMeter()
    meter.update(2.0, n=4)
    meter.update(4.0, n=2)
    print(f"AverageMeter test: avg={meter.avg:.3f}")  # expect 2.667

    # Smoke test train_one_epoch() on a tiny CNN + tiny data subset.
    from src.data import get_dataloaders
    from src.models import build_model

    loaders = get_dataloaders(subset_fraction=0.02, image_size=224, batch_size=8, num_workers=0)
    model = build_model("cnn", num_classes=10).to(device)

    history = train_model("cnn", model, loaders, device, num_epochs=2)
    print(f"Training history: {history}")
