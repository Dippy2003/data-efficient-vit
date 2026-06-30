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


def get_attention_maps(vit_model, image_tensor):
    """
    Extract the self-attention weights from the last transformer block of a
    timm ViT model for a single image.

    How ViT attention works: after splitting the image into patches, each
    patch attends to every other patch via a learned attention weight. In a
    well-trained model, the patch containing the main object tends to have
    high attention weights to other informative patches -- so the attention
    map acts like a rough "where is the model looking?" visualisation.

    Technical note: timm ViTs expose their blocks as model.blocks[i]. Each
    block's attn module has an attn_drop submodule but we hook the softmax
    output directly by temporarily registering a forward hook. We use the
    *last* block because its attention typically captures the most
    semantically meaningful relationships (earlier blocks tend to attend
    more locally, like edges and textures).

    Parameters
    ----------
    vit_model : nn.Module  (a timm vit_tiny_patch16_224, already eval())
    image_tensor : torch.Tensor  shape (1, 3, H, W), already on the right device

    Returns
    -------
    numpy array of shape (num_heads, num_patches+1, num_patches+1)
    The +1 is the [CLS] token. Index [h, 0, 1:] gives the CLS token's
    attention to each patch for head h -- that's what we'll plot.
    """
    import torch

    attention_output = []

    def hook_fn(module, input, output):
        # timm's Attention.forward returns the attended values, not the
        # weights. We need to re-extract the weights from inside the module.
        # timm stores Q,K,V in a single fused linear layer (qkv).
        B, N, C = input[0].shape
        qkv = module.qkv(input[0])
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        scale = (C // module.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        attention_output.append(attn.detach().cpu().numpy())

    last_block = vit_model.blocks[-1]
    handle = last_block.attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        vit_model(image_tensor)

    handle.remove()

    # Shape: (1, num_heads, num_patches+1, num_patches+1) -> drop batch dim
    return attention_output[0][0]


def plot_attention_overlay(
    vit_model, image_tensor, class_name: str = "", model_name: str = "vit_pretrained",
    save_path: str = None
) -> str:
    """
    Overlay the ViT's mean head attention map on the original image and save
    the result as a PNG side-by-side comparison (original | attention | overlay).

    Why attention maps matter for this project: they let you *show* what the
    model has learned. A well-trained pretrained ViT will typically highlight
    the object (cat, car, etc.) rather than the background -- even though the
    model was never explicitly told which pixels matter. A from-scratch ViT
    trained on a tiny dataset usually shows a much noisier, diffuse pattern,
    visually reinforcing why it underperforms.

    Parameters
    ----------
    vit_model : nn.Module  (a timm ViT, already eval() and on device)
    image_tensor : torch.Tensor  shape (1, 3, 224, 224)
    class_name : str  used in the figure title (e.g. "cat")
    model_name : str  used in the filename
    save_path : str, optional

    Returns
    -------
    str : path the figure was saved to.
    """
    from PIL import Image as PILImage
    import torch.nn.functional as F

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, f"{model_name}_attention.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    attn = get_attention_maps(vit_model, image_tensor)  # (heads, 197, 197)

    # CLS token (row 0) attention to all patch tokens (columns 1:)
    # Average across heads to get a single map
    cls_attn = attn[:, 0, 1:]          # (heads, 196)
    mean_attn = cls_attn.mean(axis=0)  # (196,)

    # Reshape to spatial grid: 196 = 14x14 patches (for 224/16)
    grid_size = int(mean_attn.shape[0] ** 0.5)
    attn_map = mean_attn.reshape(grid_size, grid_size)

    # Normalise to [0, 1]
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Upsample from 14x14 to 224x224 using bilinear interpolation
    import torch
    attn_tensor = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()
    attn_upsampled = F.interpolate(attn_tensor, size=(224, 224), mode="bilinear",
                                   align_corners=False).squeeze().numpy()

    # Denormalise image from model-normalised tensor to [0,1] RGB
    CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
    CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616])
    img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * CIFAR10_STD + CIFAR10_MEAN, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(attn_upsampled, cmap="hot")
    axes[1].set_title("Attention map (CLS → patches)")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(attn_upsampled, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    title = f"Attention: {model_name}"
    if class_name:
        title += f" — class: {class_name}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] saved attention overlay to {save_path}")
    return save_path
