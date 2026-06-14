"""
Configuration for RNN rumor detection model.
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
EMBEDDING_DIM = 100      # Word embedding dimension
HIDDEN_DIM = 128         # GRU hidden state dimension
MAX_LEN = 64             # Max token length per text
BATCH_SIZE = 32          # Training batch size
EPOCHS = 10              # Number of training epochs
LEARNING_RATE = 1e-3     # Adam learning rate
WEIGHT_DECAY = 1e-5      # L2 regularization
MIN_WORD_FREQ = 2        # Minimum word frequency for vocabulary

# ================ Device ================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================ Random Seed ================
SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
