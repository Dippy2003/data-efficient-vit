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
