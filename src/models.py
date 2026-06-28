"""
Model builders for the Data-Efficient ViT project.

This module is the single place that constructs the three models we compare:
  1. build_vit_scratch()      -- ViT trained from scratch (no pretraining)
  2. build_cnn()              -- ResNet-18 CNN baseline
  3. build_vit_pretrained()   -- ImageNet-pretrained ViT, fine-tuned

Keeping all three builders here (rather than scattered across notebook
cells) means the training/evaluation code in later modules can stay
model-agnostic: it just calls build_model(name) and gets back something
with a standard nn.Module interface (forward(images) -> logits).
"""

import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.

    Why this matters for this project: part of the "ViTs need more data"
    story is comparing model capacity. A from-scratch ViT and a ResNet-18
    might have a similar number of parameters, yet the ViT performs worse
    on a small dataset -- the gap is about *inductive bias*, not raw
    capacity. This function gives the numbers you'd cite to make that case.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict with keys "total" and "trainable" (both int, parameter counts).
    "trainable" can be less than "total" when some layers are frozen, e.g.
    if you choose to freeze the backbone of the pretrained ViT.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # Placeholder smoke test -- will exercise real builders once they exist.
    print("src/models.py loaded successfully. Builders coming next.")
