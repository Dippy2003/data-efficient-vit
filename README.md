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
  data.py             # dataset loading, transforms, dataloaders
  models.py           # model builders (ViT-scratch, CNN, pretrained ViT)
  train.py            # training loop, optimizer/scheduler, checkpointing
  evaluate.py         # accuracy, confusion matrix, per-class metrics (coming next)
  visualize.py        # loss/accuracy curves, attention-map overlays (coming next)
  test_integration.py # smoke tests: data + models + training all wired together
notebooks/
  main.ipynb          # end-to-end walkthrough with explanations
outputs/
  figures/            # saved plots (gitignored, regenerate by running code)
  checkpoints/        # saved model weights (gitignored, regenerate by training)
  history/            # per-model training curves as JSON (gitignored, regenerate by training)
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

## Training

`src/train.py` provides `train_model(model_name, model, loaders, device,
num_epochs)`, which works the same way for all 3 models (CNN, ViT-scratch,
ViT-pretrained). It logs train/val loss and accuracy per epoch, saves the
best checkpoint to `outputs/checkpoints/`, and saves the full training
history to `outputs/history/<model_name>_history.json` for later plotting.

Two presets are defined at the top of `src/train.py`:

- `DEV_CONFIG` — tiny data subset, few epochs, for fast iteration while
  writing code. Results are not meaningful, just useful for catching bugs.
- `FINAL_CONFIG` — full dataset, more epochs, for the run whose numbers
  you'd report. This takes noticeably longer, especially for the ViTs —
  a GPU is strongly recommended (Colab's free GPU runtime works fine).

Quick manual check: `python -m src.train` runs a 2-epoch CNN training pass
on a tiny subset and confirms checkpoint save/load works. For a check that
trains all 3 models together, run `python -m src.test_integration`.

### Optimizer choices

All 3 models use AdamW with a cosine learning-rate schedule, but the
learning rate differs by model (`get_optimizer()` in `src/train.py`):

| Model | LR | Why |
|---|---|---|
| `vit_scratch` | 1e-3 | must learn everything from zero, so it needs bigger steps |
| `cnn` | 1e-3 | same reasoning — random init, no pretraining |
| `vit_pretrained` | 1e-4 | fine-tuning should nudge pretrained weights gently, not overwrite them |

### Model sizes

| Model | Parameters |
|---|---|
| `vit_scratch` (ViT-Tiny) | ~5.5M |
| `cnn` (ResNet-18) | ~11.2M |
| `vit_pretrained` (ViT-Tiny) | ~5.5M |

Note the from-scratch ViT is actually *smaller* than the CNN baseline, which
is useful for the project's argument: any accuracy gap is about
architecture/inductive bias, not raw model capacity.
