"""
公共配置（供逻辑回归模型使用）
"""

import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
