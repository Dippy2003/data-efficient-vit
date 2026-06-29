"""
Data pipeline for the Data-Efficient ViT project.

This module is the *only* place that should know about a specific dataset
(CIFAR-10 for now). Everything downstream (models, training, evaluation)
just consumes PyTorch DataLoaders, so swapping CIFAR-10 for a plant-disease
dataset or MedMNIST later only means adding a branch in `get_dataset()` --
the rest of the codebase does not change.

Why augmentation matters for this project specifically: the whole point of
the experiment is to show that ViTs need more data than CNNs to reach the
same accuracy, because they lack convolution's built-in assumptions about
images (nearby pixels are related, and shifting an object shouldn't change
the label). Augmentation (random crops/flips) is a cheap way to synthetically
grow a small dataset, so it's worth keeping consistent across all 3 models
to make the comparison fair.
"""

import warnings

import torch
from torch.utils.data import DataLoader, Subset

# torchvision's CIFAR-10 loader triggers a harmless NumPy 2.4 deprecation
# warning internally (unrelated to our code -- it's in torchvision's pickle
# loading path). Silencing just this one warning keeps real warnings visible.
warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed")
import torchvision
import torchvision.transforms as T

# CIFAR-10 per-channel mean/std, computed over the training set.
# Normalizing with these (rather than generic 0.5/0.5/0.5) gives slightly
# faster, more stable convergence -- a standard practice, not something
# specific to ViTs or CNNs.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_transforms(image_size: int = 224):
    """
    Build the train and eval transform pipelines.

    Parameters
    ----------
    image_size : int
        Final image size fed to the model. We resize up to 224x224 (instead
        of CIFAR-10's native 32x32) because both `timm` ViT models and
        torchvision's ResNet-18 expect inputs close to ImageNet's 224x224 --
        this matters especially for the *pretrained* ViT, whose learned
        position embeddings assume that resolution.

    Returns
    -------
    (train_transform, eval_transform) : tuple of torchvision.transforms.Compose

    Notes
    -----
    - train_transform includes RandomCrop-after-pad and RandomHorizontalFlip:
      standard, label-preserving augmentations for natural images like
      CIFAR-10 (flipping a cat horizontally is still a cat).
    - eval_transform has NO augmentation -- validation/test accuracy must
      reflect the model's real performance, not a randomly perturbed image.
    """
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomCrop(image_size, padding=image_size // 8, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, eval_transform


def get_dataset(name: str, root: str, train: bool, transform):
    """
    Dataset factory. Add new datasets here as branches -- this is the single
    point of change needed to plug in, e.g., a plant-disease dataset later.

    Parameters
    ----------
    name : str
        Dataset identifier, e.g. "cifar10".
    root : str
        Directory to download/read data from.
    train : bool
        Whether to load the train or test split.
    transform : callable
        torchvision transform to apply to each image.
    """
    name = name.lower()
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
    raise ValueError(f"Unknown dataset '{name}'. Add it to get_dataset() in src/data.py.")


def get_dataloaders(
    dataset_name: str = "cifar10",
    data_root: str = "./data",
    image_size: int = 224,
    batch_size: int = 64,
    subset_fraction: float = 1.0,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
):
    """
    Build train / val / test DataLoaders for the chosen dataset.

    Parameters
    ----------
    dataset_name : str
        Passed straight to get_dataset(). Default "cifar10".
    data_root : str
        Where to store/download the raw dataset files.
    image_size : int
        Side length images are resized to before going into a model.
    batch_size : int
        Samples per batch.
    subset_fraction : float in (0, 1]
        Use only this fraction of the *training* data. This is the dial for
        fast development iterations -- e.g. 0.1 trains on 10% of CIFAR-10 so
        a full epoch takes a fraction of the time. Set to 1.0 for the final,
        reported results.
    val_fraction : float in (0, 1)
        Fraction of the (possibly subsetted) training data held out for
        validation, used to track over/underfitting during training.
    num_workers : int
        Parallel data-loading worker processes. On Windows/Colab, 2 is a
        safe default; set to 0 if you hit multiprocessing errors.
    seed : int
        Random seed for the subset/val split, so results are reproducible.

    Returns
    -------
    dict with keys "train", "val", "test" -> DataLoader, and "classes" ->
    tuple of class names.
    """
    if not (0 < subset_fraction <= 1.0):
        raise ValueError("subset_fraction must be in (0, 1].")
    if not (0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1).")

    train_transform, eval_transform = get_transforms(image_size)

    # Two separate dataset objects over the same train split, one with
    # augmentation (for training) and one without (for validation) --
    # torchvision re-reads from disk/cache so this has no extra download cost.
    full_train_aug = get_dataset(dataset_name, data_root, train=True, transform=train_transform)
    full_train_eval = get_dataset(dataset_name, data_root, train=True, transform=eval_transform)
    test_set = get_dataset(dataset_name, data_root, train=False, transform=eval_transform)

    n_total = len(full_train_aug)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()

    n_used = int(n_total * subset_fraction)
    used_indices = perm[:n_used]

    n_val = int(n_used * val_fraction)
    val_indices = used_indices[:n_val]
    train_indices = used_indices[n_val:]

    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    classes = CIFAR10_CLASSES if dataset_name.lower() == "cifar10" else None

    print(
        f"[data] dataset={dataset_name} subset_fraction={subset_fraction} "
        f"-> train={len(train_set)} val={len(val_set)} test={len(test_set)} "
        f"(image_size={image_size}, batch_size={batch_size})"
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "classes": classes,
    }


if __name__ == "__main__":
    # Quick manual smoke test: run `python src/data.py` to sanity-check the
    # pipeline without needing a notebook. Uses a small subset and small
    # image size so it finishes fast even on CPU.
    loaders = get_dataloaders(subset_fraction=0.05, image_size=64, batch_size=8, num_workers=0)
    images, labels = next(iter(loaders["train"]))
    print(f"Batch shape: {images.shape}, labels: {labels[:8].tolist()}")
    print(f"Classes: {loaders['classes']}")
