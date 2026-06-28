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


if __name__ == "__main__":
    main()
