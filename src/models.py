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

import timm
import torch.nn as nn
import torchvision.models as tv_models


def build_vit_scratch(num_classes: int = 10, img_size: int = 224) -> nn.Module:
    """
    Build a Vision Transformer with NO pretrained weights -- it learns
    everything, including basic visual structure, from our small dataset.

    Why this is expected to underperform: a ViT first chops the image into
    fixed-size patches (here 16x16) and treats them like a sequence of
    "tokens", then uses self-attention to relate patches to each other. Self
    -attention has no built-in notion that nearby pixels are usually related
    (locality) or that shifting an object shouldn't change the label
    (translation invariance) -- a CNN gets both for free from how
    convolution kernels work. A ViT *can* learn these patterns, but only by
    seeing enough examples; with a small dataset like a subset of CIFAR-10,
    there typically isn't enough signal, so we expect this model to lag
    behind the CNN baseline.

    Parameters
    ----------
    num_classes : int
        Number of output classes (10 for CIFAR-10).
    img_size : int
        Input image side length. Must match the size used by the data
        pipeline (src/data.py uses 224 by default).

    Returns
    -------
    nn.Module
        A `vit_tiny_patch16_224` model (timm) with random initialization.
        "Tiny" is chosen deliberately: it is the smallest standard ViT
        variant, which keeps training time reasonable on a small dataset
        and on modest hardware (including Colab's free GPU tier).
    """
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size,
    )
    return model


def build_cnn(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Build a ResNet-18 CNN baseline.

    Parameters
    ----------
    num_classes : int
        Number of output classes (10 for CIFAR-10).
    pretrained : bool
        If False (the default for this project), weights are randomly
        initialized -- the CNN trains from scratch on the small dataset,
        just like the from-scratch ViT. This keeps the comparison fair:
        any performance gap between the CNN and ViT-scratch is then
        attributable to architecture (convolutional inductive bias), not
        to one model getting a head start from pretraining.
        Set True only if you want a pretrained-CNN side-experiment.

    Returns
    -------
    nn.Module
        ResNet-18 with its final fully-connected layer replaced to output
        `num_classes` logits instead of ImageNet's 1000.
    """
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet18(weights=weights)

    # ResNet-18's classifier head is `model.fc`, a single Linear layer.
    # We replace it so the output dimension matches our number of classes.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


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
    import torch

    dummy_images = torch.randn(2, 3, 224, 224)  # batch of 2 fake RGB images

    cnn = build_cnn(num_classes=10, pretrained=False)
    logits = cnn(dummy_images)
    print(f"CNN output shape: {logits.shape}")  # expect [2, 10]
    print(f"CNN parameters: {count_parameters(cnn)}")

    vit_scratch = build_vit_scratch(num_classes=10, img_size=224)
    logits = vit_scratch(dummy_images)
    print(f"ViT-scratch output shape: {logits.shape}")  # expect [2, 10]
    print(f"ViT-scratch parameters: {count_parameters(vit_scratch)}")
