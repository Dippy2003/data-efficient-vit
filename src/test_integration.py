"""
Integration smoke test: confirms the data pipeline (src/data.py) and the
model builders (src/models.py) actually work together end-to-end.

Run with: python src/test_integration.py

This isn't a unit test framework (pytest etc.) -- just a fast, runnable
script you can call before each training run to catch shape mismatches or
broken imports early, without waiting for a full training loop to fail.
"""

from src.data import get_dataloaders, CIFAR10_CLASSES
from src.evaluate import (
    compute_accuracy, confusion_matrix_from_loader,
    per_class_report, build_results_table, print_results_table,
)
from src.models import MODEL_REGISTRY, build_model, count_parameters
from src.train import get_device, load_checkpoint, train_model
from src.visualize import (
    plot_training_curves, plot_confusion_matrix,
    plot_attention_overlay, plot_sample_predictions,
)


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


def test_evaluate_and_visualize():
    """
    Confirms evaluate.py and visualize.py work end-to-end using the
    checkpoints saved by the earlier training test.
    """
    device = get_device()
    loaders = get_dataloaders(
        subset_fraction=0.02, image_size=224, batch_size=8, num_workers=0
    )

    # Load all 3 checkpoints
    models = {}
    for name in MODEL_REGISTRY:
        m = build_model(name).to(device)
        models[name] = load_checkpoint(m, name, device)

    # Accuracy
    for name, model in models.items():
        acc = compute_accuracy(model, loaders["test"], device)
        print(f"{name} test accuracy: {acc:.4f}")

    # Confusion matrix + plot for CNN
    cm, classes = confusion_matrix_from_loader(models["cnn"], loaders["test"], device, CIFAR10_CLASSES)
    plot_confusion_matrix(cm, classes, "cnn")
    print("Confusion matrix plot: OK")

    # Per-class report
    report = per_class_report(models["cnn"], loaders["test"], device, CIFAR10_CLASSES)
    assert "precision" in report
    print("Per-class report: OK")

    # Results table
    rows = build_results_table(models, loaders["test"], device)
    print_results_table(rows)

    # Training curves (uses saved JSON histories)
    plot_training_curves()
    print("Training curves plot: OK")

    # Sample predictions
    plot_sample_predictions(models["cnn"], loaders["test"], device, CIFAR10_CLASSES, "cnn")
    print("Sample predictions plot: OK")

    # Attention map on pretrained ViT
    images, labels = next(iter(loaders["test"]))
    plot_attention_overlay(
        models["vit_pretrained"], images[:1].to(device),
        class_name=CIFAR10_CLASSES[labels[0].item()]
    )
    print("Attention overlay plot: OK")

    print("Integration test passed: evaluate + visualize pipeline works end-to-end.")


if __name__ == "__main__":
    main()
    test_train_all_models()
    test_evaluate_and_visualize()
