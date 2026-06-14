"""
Configuration for TextCNN + GloVe rumor detection model.
"""

import os
import torch

# ================ Paths ================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
GLOVE_DIR = os.path.join(os.path.dirname(BASE_DIR), "glove")

MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "textcnn_glove.pt")
VOCAB_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "vocab_textcnn.pt")

# ================ GloVe ================
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.200d.txt"        # 200-dim vectors, 400K words
GLOVE_DIM = 200
GLOVE_FREEZE = False                     # fine-tune embeddings during training

# ================ TextCNN Hyperparameters ================
EMBEDDING_DIM = GLOVE_DIM               # must match GloVe
FILTER_SIZES = (2, 3, 4, 5)            # 2~5 gram convolutions
N_FILTERS = 256                         # ↑ 128→256, more pattern capacity
DROPOUT = 0.4                           # grid search optimal
NUM_LAYERS = 1
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 2e-4
MIN_WORD_FREQ = 2                       # filter rare words → higher GloVe coverage

# ================ LR Scheduler ================
LR_PATIENCE = 3
LR_FACTOR = 0.5

# ================ Early Stopping ================
EARLY_STOP_PATIENCE = 7

# ================ CPU Optimization ================
CPU_THREADS = 10
DATALOADER_WORKERS = 0

# ================ Device ================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================ Random Seed ================
SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GLOVE_DIR, exist_ok=True)
