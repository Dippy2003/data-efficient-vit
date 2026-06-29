"""
Training loop for the Data-Efficient ViT project.

This module trains any of the 3 models from src/models.py using the
dataloaders from src/data.py. Kept model-agnostic on purpose: it just calls
model(images) and expects logits back, so the same training code works for
the CNN and both ViT variants without special-casing.
"""

import torch


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


if __name__ == "__main__":
    device = get_device()
    print(f"Selected device: {device}")

    meter = AverageMeter()
    meter.update(2.0, n=4)
    meter.update(4.0, n=2)
    print(f"AverageMeter test: avg={meter.avg:.3f}")  # expect 2.667
