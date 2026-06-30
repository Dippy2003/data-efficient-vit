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
  evaluate.py         # accuracy, confusion matrix, per-class metrics, results table
  visualize.py        # training curves, confusion heatmap, attention maps, sample predictions
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

## Evaluation

`src/evaluate.py` provides four functions, all usable after loading a
checkpoint with `load_checkpoint()` from `src/train.py`:

| Function | What it produces |
|---|---|
| `compute_accuracy()` | Top-1 accuracy (single float) on any loader |
| `confusion_matrix_from_loader()` | NxN matrix of true vs predicted class counts |
| `per_class_report()` | Precision, recall, F1 per class (sklearn formatted string) |
| `build_results_table()` | List of rows comparing all 3 models: accuracy + macro F1 |
| `print_results_table()` | Pretty-prints the above table and saves to `outputs/results_table.txt` |

Expected results (1 epoch, 2% of CIFAR-10 — indicative only):

```
Model                  Accuracy   Macro F1
------------------------------------------
vit_scratch              0.172      0.077
cnn                      0.201      0.132
vit_pretrained           0.696      0.692
```

With the full dataset and more epochs, all three improve, but the ranking
stays the same — confirming the data-efficiency story.

## Visualization

`src/visualize.py` saves all plots to `outputs/figures/`:

| Function | Output file |
|---|---|
| `plot_training_curves()` | `training_curves.png` — loss and accuracy per epoch for all 3 models |
| `plot_confusion_matrix()` | `<model>_confusion.png` — normalised confusion heatmap |
| `plot_sample_predictions()` | `<model>_predictions.png` — grid of images, green=correct, red=wrong |
| `plot_attention_overlay()` | `<model>_attention.png` — ViT attention overlaid on a sample image |

Attention maps work on both ViT variants. A pretrained ViT's attention is
usually focused on the object; a from-scratch ViT's attention is noisier,
which is another way to visualise the data-efficiency gap beyond accuracy.
