"""
Training loop for the Data-Efficient ViT project.

This module trains any of the 3 models from src/models.py using the
dataloaders from src/data.py. Kept model-agnostic on purpose: it just calls
model(images) and expects logits back, so the same training code works for
the CNN and both ViT variants without special-casing.
"""

import torch


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
