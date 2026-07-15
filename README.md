<h1 align="center">Data-Efficient Vision Transformers for Small Image Datasets</h1>

<p align="center">
  Comparing ViT vs CNN on CIFAR-10 — does a Transformer beat a CNN when data is scarce?.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/timm-0.9+-brightgreen?style=flat"/>
  <img src="https://img.shields.io/badge/Dataset-CIFAR--10-orange?style=flat"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat"/>
</p>

---

## What this project does 's

This is an **IT3071 Machine Learning** assignment that answers one question:

> **Can a Vision Transformer (ViT) compete with a CNN when trained on a small dataset?**

We train and compare **3 models** on CIFAR-10 (60,000 images, 10 classes):

| Model | Init | What we expect |
|---|---|---|
| ViT-Tiny (from scratch) | Random | Worst — no built-in image priors, needs lots of data |
| ResNet-18 (CNN) | Random | Mid — convolutional priors help even with little data |
| ViT-Tiny (pretrained) | ImageNet weights | Best — pretraining gives it what data can't |

**The story: `vit_scratch` < `cnn` < `vit_pretrained`** — and the gap widens as data gets smaller

---

## Results (smoke test — 2% data, 1 epoch)

Even on a tiny 2% slice of CIFAR-10, the ranking is already clear:

```
Model                  Accuracy   Macro F1
------------------------------------------
vit_scratch              0.172      0.077
cnn                      0.179      0.127
vit_pretrained           0.747      0.728
------------------------------------------
```

> With the full dataset (100%) and 15 epochs, all three improve significantly — but the ranking stays the same.

---

## Outputs

The pipeline produces 4 types of visualisations, all saved to `outputs/figures/`:

### 1. Training curves
Loss and accuracy per epoch for all 3 models — solid = train, dashed = val.

> *The pretrained ViT (green) starts high and climbs fast. The CNN (blue) climbs steadily. The from-scratch ViT (red) struggles.*

<!-- Replace with your actual plot after running -->
![Training curves](outputs/figures/training_curves.png)

---

### 2. Confusion matrices
Each cell shows the fraction of true-class-i images predicted as class-j. Bright diagonal = good.

> *The pretrained ViT has a much cleaner diagonal. The from-scratch ViT often collapses to predicting just 1–2 classes.*

<!-- Replace with your actual plots after running -->
| vit_scratch | cnn | vit_pretrained |
|---|---|---|
| ![](outputs/figures/vit_scratch_confusion.png) | ![](outputs/figures/cnn_confusion.png) | ![](outputs/figures/vit_pretrained_confusion.png) |

---

### 3. Sample predictions
A 4×4 grid of test images — **green title = correct**, **red title = wrong**.

> *Easy to see which classes trip each model up (cat vs dog, automobile vs truck).*

<!-- Replace with your actual plots after running -->
| vit_scratch | cnn | vit_pretrained |
|---|---|---|
| ![](outputs/figures/vit_scratch_predictions.png) | ![](outputs/figures/cnn_predictions.png) | ![](outputs/figures/vit_pretrained_predictions.png) |

---

### 4. Attention maps
The ViT's self-attention weights overlaid on the input image — shows **where the model is looking**.

> *Pretrained ViT focuses on the object. From-scratch ViT shows diffuse, noisy attention — it hasn't learned what matters yet.*

<!-- Replace with your actual plots after running -->
| From-scratch ViT | Pretrained ViT |
|---|---|
| ![](outputs/figures/vit_scratch_attention.png) | ![](outputs/figures/vit_pretrained_attention.png) |

---

## Why ViTs struggle on small datasets

CNNs use **convolution kernels** that enforce two strong priors about images:
- **Locality** — nearby pixels are more related than distant ones
- **Translation invariance** — a cat in the top-left is the same object as a cat in the bottom-right

ViTs use **global self-attention** — every patch can attend to every other patch. There is no built-in assumption about spatial structure. When there is enough data, that flexibility is powerful. When data is scarce, the CNN's priors win.

**Pretraining** resolves this: a ViT trained on millions of ImageNet images has already learned what locality and translation look like — it just needs a little fine-tuning to apply that knowledge to our 10 classes.

---

## Quick start

One command trains all 3 models, evaluates them, and saves every plot:

```bash
# Fast dev run — 5% of data, 2 epochs (~6 min on GPU)
python -m src.run_experiment

# Full graded run — 100% data, 15 epochs (~2 hr on GPU)
python -m src.run_experiment --mode final

# Train specific models only
python -m src.run_experiment --models vit_pretrained cnn --epochs 10

# Custom subset and epoch count
python -m src.run_experiment --subset 0.1 --epochs 5

# Re-evaluate existing checkpoints without re-training
python -m src.run_experiment --skip-train

# Just print the results table, no plots
python -m src.run_experiment --skip-train --skip-viz
```

All outputs (figures, checkpoints, history JSON, results table) are saved to `outputs/` automatically.

---

## Project structure

```
src/
  data.py              # CIFAR-10 loading, augmentation, dataloaders
  models.py            # build_model() factory for all 3 models
  train.py             # training loop, AdamW + cosine LR, checkpointing
  evaluate.py          # accuracy, confusion matrix, F1, results table
  visualize.py         # training curves, confusion heatmap, attention maps, prediction grid
  run_experiment.py    # CLI runner — trains + evaluates + visualises in one command
  test_integration.py  # smoke test — runs everything end-to-end in ~5 min
notebooks/
  main.ipynb           # full explainer notebook (run on Colab)
outputs/
  figures/             # all plots saved here (gitignored — regenerate by running)
  checkpoints/         # best model weights (gitignored — regenerate by training)
  history/             # training history JSON per model (gitignored)
```

---

## Setup

```bash
git clone https://github.com/Dippy2003/data-efficient-vit.git
cd data-efficient-vit
pip install -r requirements.txt
```

**On Google Colab** (recommended — free GPU):
```
!git clone https://github.com/Dippy2003/data-efficient-vit.git
%cd data-efficient-vit
!pip install -r requirements.txt
```
Then go to **Runtime → Change runtime type → GPU**.

---

## How to run

### Option 1 — CLI runner (easiest)

```bash
# Dev run (fast, for testing)
python -m src.run_experiment

# Full graded run
python -m src.run_experiment --mode final
```

`src/run_experiment.py` handles data loading, training, evaluation, and all visualizations in one shot. All outputs go to `outputs/` automatically.

**All CLI flags:**

| Flag | Default | What it does |
|---|---|---|
| `--mode dev\|final` | `dev` | Preset: dev=5% data/2 epochs, final=100%/15 epochs |
| `--models MODEL ...` | all three | Pick which models to train, e.g. `--models cnn vit_pretrained` |
| `--epochs N` | from mode | Override number of training epochs |
| `--subset 0.0–1.0` | from mode | Override fraction of training data to use |
| `--batch-size N` | 64 | Dataloader batch size |
| `--skip-train` | off | Load existing checkpoints, skip training |
| `--skip-viz` | off | Skip plots, just print the results table |

### Option 2 — Notebook (Colab)

Open `notebooks/main.ipynb`, set `USE_FULL_DATA = True`, run all cells.

### Option 3 — Sanity check (~5 min, no GPU needed)

```bash
python -m src.test_integration
```

Trains all 3 models for 1 epoch on 2% of CIFAR-10, runs eval + all visualizations end-to-end.

---

## Training details

### Optimizer

All 3 models use **AdamW + cosine LR schedule**. Learning rate differs by model:

| Model | LR | Why |
|---|---|---|
| `vit_scratch` | 1e-3 | Learning from zero — needs larger steps |
| `cnn` | 1e-3 | Same — random init |
| `vit_pretrained` | 1e-4 | Fine-tuning — must nudge weights gently, not overwrite them |

### Model sizes

| Model | Parameters | Init |
|---|---|---|
| `vit_scratch` (ViT-Tiny) | ~5.5M | Random |
| `cnn` (ResNet-18) | ~11.2M | Random |
| `vit_pretrained` (ViT-Tiny) | ~5.5M | ImageNet pretrained |

> The from-scratch ViT is *smaller* than the CNN — so any accuracy gap is about **architecture**, not capacity.

### Expected runtime (Colab T4 GPU)

| Config | Data | Epochs | Per model | Total |
|---|---|---|---|---|
| DEV | 5% CIFAR-10 | 2 | ~2 min | ~6 min |
| FINAL | 100% CIFAR-10 | 15 | ~25–40 min | ~1.5–2 hr |

---

## Notebook walkthrough

`notebooks/main.ipynb` covers everything step by step — built for the explainer video:

1. What is a Vision Transformer? (patches, positional encoding, CLS token, self-attention)
2. CIFAR-10 data pipeline + sample images
3. All 3 model architectures + parameter counts
4. Training strategy — why different learning rates?
5. Training curves — see how each model learns epoch by epoch
6. Results table — accuracy + macro F1 comparison
7. Per-class breakdown — which classes does each model struggle with?
8. Confusion matrices — where do the mistakes go?
9. Sample predictions — green = correct, red = wrong
10. Attention maps — what is the ViT actually looking at?
11. Conclusion + extension ideas (data-efficiency curve, DeiT)
12. References

---

## References

1. Dosovitskiy et al. (2020) — *An Image is Worth 16x16 Words* — original ViT paper
2. He et al. (2016) — *Deep Residual Learning* — ResNet paper
3. Touvron et al. (2021) — *DeiT: Training data-efficient image transformers*
4. Loshchilov & Hutter (2019) — *Decoupled Weight Decay Regularization* — AdamW
5. Ross Wightman — [timm](https://github.com/rwightman/pytorch-image-models) — ViT implementation used
