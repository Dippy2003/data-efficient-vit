# Data-Efficient Vision Transformers for Small Image Datasets

IT3071 Machine Learning project. Compares three models on a small image
dataset to demonstrate the data-efficiency gap between Vision Transformers
(ViTs) and CNNs:

1. **ViT trained from scratch** — expected to underperform, since it lacks
   the convolutional inductive biases (locality, translation invariance)
   that let CNNs learn well from limited data.
2. **CNN baseline (ResNet-18)** — expected to do reasonably well even with
   little data, due to those built-in biases.
3. **Pretrained ViT, fine-tuned** — expected to perform best, since
   large-scale pretraining (e.g. on ImageNet) gives it the visual
   representations it can't learn from a small dataset alone.

Expected story: **from-scratch ViT < CNN < fine-tuned ViT**.

## Project structure

```
src/
  data.py        # dataset loading, transforms, dataloaders
  models.py      # model builders (ViT-scratch, CNN, pretrained ViT)
  train.py       # training loop
  evaluate.py    # accuracy, confusion matrix, per-class metrics
  visualize.py   # loss/accuracy curves, attention-map overlays
notebooks/
  main.ipynb     # end-to-end walkthrough with explanations
outputs/
  figures/       # saved plots
  checkpoints/   # saved model weights (gitignored, regenerate by training)
```

## Setup

```bash
pip install -r requirements.txt
```

On Google Colab, just `!pip install -r requirements.txt` in a cell — a GPU
runtime is recommended (Runtime > Change runtime type > GPU) since
training a ViT from scratch is slow on CPU.

## Dataset

Starts with CIFAR-10 (auto-downloaded by torchvision). The data pipeline in
`src/data.py` is written so other small image datasets (e.g. a plant-disease
dataset, MedMNIST) can be swapped in later without changing the rest of the
code.
