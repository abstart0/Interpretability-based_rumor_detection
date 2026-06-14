"""
Training and evaluation pipeline for the Logistic Regression rumor detection model.
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 允许从任何目录作为脚本运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import TRAIN_PATH, VAL_PATH, CHECKPOINT_DIR, SEED
from models.lr_model import RumorLR

# 定义逻辑回归模型特有的保存路径，防止覆盖同学的权重
LR_MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "lr_model.pkl")

def set_seed(seed: int):
    """固定随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)

def train():
    """逻辑回归主训练流程"""
    set_seed(SEED)

    print(f"Loading data from: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # ---------- 初始化模型 ----------
    lr_classifier = RumorLR()

    print("Preprocessing text...")
    X_train = train_df['text'].apply(lr_classifier.preprocess)
    X_val = val_df['text'].apply(lr_classifier.preprocess)
    y_train = train_df['label']
    y_val = val_df['label']

    # ---------- 特征提取与模型训练 ----------
    print("Extracting TF-IDF features and training Logistic Regression...")
    X_train_vec = lr_classifier.vectorizer.fit_transform(X_train)
    X_val_vec = lr_classifier.vectorizer.transform(X_val)

    lr_classifier.model.fit(X_train_vec, y_train)

    # ---------- 验证集评估 ----------
    val_preds = lr_classifier.model.predict(X_val_vec)
    
    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds, zero_division=0)
    rec = recall_score(y_val, val_preds, zero_division=0)
    f1 = f1_score(y_val, val_preds, zero_division=0)

    print(f"\n{'='*20} LR Evaluation {'='*20}")
    print(f"Val Accuracy:  {acc:.4f}")
    print(f"Val Precision: {prec:.4f}")
    print(f"Val Recall:    {rec:.4f}")
    print(f"Val F1:        {f1:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(y_val, val_preds, target_names=["Non-rumor", "Rumor"], digits=4))

    # ---------- 保存训练好的模型 ----------
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    joblib.dump({
        'model': lr_classifier.model,
        'vectorizer': lr_classifier.vectorizer
    }, LR_MODEL_SAVE_PATH)
    print(f"\nModel successfully saved to: {LR_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()