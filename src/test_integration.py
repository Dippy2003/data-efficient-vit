"""
Integration smoke test: confirms the data pipeline (src/data.py) and the
model builders (src/models.py) actually work together end-to-end.

Run with: python src/test_integration.py

This isn't a unit test framework (pytest etc.) -- just a fast, runnable
script you can call before each training run to catch shape mismatches or
broken imports early, without waiting for a full training loop to fail.
"""

from src.data import get_dataloaders
from src.models import MODEL_REGISTRY, build_model, count_parameters
from src.train import get_device, train_model


def main():
    # Tiny subset, tiny batch -- this only needs to prove the plumbing
    # works, not produce meaningful accuracy.
    loaders = get_dataloaders(
        subset_fraction=0.02, image_size=224, batch_size=4, num_workers=0
    )
    images, labels = next(iter(loaders["train"]))
    print(f"Batch from data pipeline: images={images.shape}, labels={labels.shape}")

    for name in MODEL_REGISTRY:
        model = build_model(name, num_classes=10, img_size=224)
        model.eval()
        logits = model(images)
        assert logits.shape == (images.shape[0], 10), (
            f"{name} produced unexpected output shape {logits.shape}"
        )
        print(f"{name}: OK, output={logits.shape}, params={count_parameters(model)['total']:,}")

    print("Integration test passed: data pipeline and all 3 models are compatible.")


def test_train_all_models():
    """
    Train each of the 3 models for 1 tiny epoch on a tiny subset. This
    confirms train_model() works uniformly across the CNN and both ViT
    variants -- catches model-specific bugs (e.g. an optimizer not liking
    a particular model's parameters) before a real, multi-hour run.
    """
    device = get_device()
    loaders = get_dataloaders(
        subset_fraction=0.02, image_size=224, batch_size=4, num_workers=0
    )

    for name in MODEL_REGISTRY:
        model = build_model(name, num_classes=10, img_size=224).to(device)
        history = train_model(name, model, loaders, device, num_epochs=1)
        assert len(history["train_loss"]) == 1
        print(f"{name}: 1-epoch training run OK")

    print("Integration test passed: all 3 models can be trained end-to-end.")


if __name__ == "__main__":
    main()
    test_train_all_models()
