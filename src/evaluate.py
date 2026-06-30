"""
Evaluation module for the Data-Efficient ViT project.

Produces three types of output for each trained model:
  1. Overall accuracy on the test set
  2. Confusion matrix (which classes get confused with which)
  3. Per-class precision, recall, F1

Having these separate from src/train.py means we can reload a saved
checkpoint and evaluate it without re-running training -- useful when
training took a long time (e.g. the pretrained ViT on the full dataset).
"""
