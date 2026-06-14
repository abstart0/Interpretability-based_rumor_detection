"""
Configuration for RNN rumor detection model.
Optimal hyperparameters from grid search.
"""

import os
import torch

# ================ Paths ================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "bigru.pt")
VOCAB_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "vocab.pt")

# ================ Hyperparameters ================
EMBEDDING_DIM = 150       # grid search: 150 > 100, 200 (balanced)
HIDDEN_DIM = 200          # grid search: 200 > 256, 128 (balanced)
NUM_LAYERS = 1            # single layer: 2 layers overfits
MAX_LEN = 64              # max text only 31 words
BATCH_SIZE = 32           # grid search: bs=32 consistently beats bs=64
EPOCHS = 20               # with early stopping
LEARNING_RATE = 5e-4      # grid search: 5e-4 beats 1e-3
WEIGHT_DECAY = 3e-4       # round 2 grid search: 3e-4 > 2e-4
MIN_WORD_FREQ = 2
DROPOUT = 0.5             # round 2 grid search: 0.5 > 0.4

# ================ LR Scheduler ================
LR_PATIENCE = 3
LR_FACTOR = 0.5

# ================ Early Stopping ================
EARLY_STOP_PATIENCE = 7

# ================ CPU Optimization ================
CPU_THREADS = 10          # Ryzen AI 9 H 365 (10 cores)
DATALOADER_WORKERS = 0    # 0 on Windows: multiprocessing kills data integrity

# ================ Device ================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================ Random Seed ================
SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
