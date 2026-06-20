"""
Shared configuration for Logistic Regression model.

Used by scripts/train_lr.py and scripts/predict_lr.py.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")

SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
