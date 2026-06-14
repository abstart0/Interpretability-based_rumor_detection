# Interpretability-based Rumor Detection

谣言检测深度学习模型 — 二分类（0=非谣言, 1=谣言）。

## 环境要求

- Python 3.11
- PyTorch 2.4.0 (或更高版本兼容版)
- 依赖见下方安装说明

## 安装

```bash
pip install pandas scikit-learn joblib torch
```

## 项目结构

```
├── predictor.py             # 统一推理接口（→ 可解释性模块调用）
├── run_textcnn.py           # 训练入口（TextCNN + GloVe）
├── run.py                   # 训练入口（BiGRU，备用）
├── config/
│   ├── textcnn_config.py    # TextCNN 超参数
│   └── config.py            # BiGRU 超参数
├── models/
│   ├── lr_model.py         # 逻辑回归基线模型
│   ├── textcnn.py           # TextCNN 模型 (Kim 2014)
│   └── bigru.py             # BiGRU 模型
├── utils/
│   ├── data_utils.py        # 分词、词表、Dataset/DataLoader
│   └── glove_utils.py       # GloVe 下载与 embedding 矩阵构建
├── scripts/
│   ├── train.py            # BiGRU 训练 + 评估流程
│   ├── predict.py          # BiGRU 推理模块 (RumorClassifier)
│   ├── train_lr.py         # 逻辑回归 训练 + 评估流程 
│   └── predict_lr.py       # 逻辑回归 推理模块 (LRRumorClassifier) 
├── data/
│   ├── train.csv           # 训练集
│   └── val.csv             # 验证集
├── checkpoints/
│   ├── bigru.pt            # 训练好的 BiGRU 模型权重
│   ├── vocab.pt            # 词表
│   └── lr_model.pkl        # 训练好的 逻辑回归 模型权重
└── interpretability/       # 可解释性模块 (待实现)
│   ├── train_textcnn.py     # TextCNN 训练脚本
│   ├── train.py             # BiGRU 训练脚本
│   └── predict.py           # BiGRU 推理（备用）
├── data/
│   ├── train.csv            # 训练集 2,840 条
│   └── val.csv              # 验证集 401 条
└── checkpoints/             # 训练产物
    ├── textcnn_glove.pt     # TextCNN 模型权重
    └── vocab_textcnn.pt     # 词表
```

## 快速开始

### 训练

首次运行自动下载 GloVe 6B 词向量（~822MB，仅一次）：

```bash
# Windows
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 py -3.11 run_textcnn.py

# Linux / macOS
OMP_NUM_THREADS=10 py -3.11 run_textcnn.py
```

### 推理

```python
from predictor import RumorDetector

detector = RumorDetector()
label, confidence = detector.predict("Breaking: scientists discover alien life!")
# label: 0 (非谣言) 或 1 (谣言)
# confidence: 0.0 ~ 1.0
```

### 训练模型

```bash
cd Interpretability-based_rumor_detection
python scripts/train_lr.py
```
### 推理预测

```python
from scripts.predict_lr import LRRumorClassifier

clf = LRRumorClassifier()
result = clf.classify("your text here")   # 0=非谣言, 1=谣言
```

## 模型性能

BiGRU
| 指标 | 数值 |
|------|------|
| Accuracy | 85.04% |
| Precision | 93.23% |
| Recall | 70.86% |
| F1 Score | 80.52% |
模型: BiGRU (embedding=100, hidden=128, vocab=2727, params=449,597)

Logistic Regression
| 指标 | 数值 |
|------|------|
| Accuracy | 82.29% |
| Precision | 84.67% |
| Recall | 72.57% |
| F1 Score | 78.15% |
模型: Logistic Regression (feature=TF-IDF, max_features=10000, params=10,001)

TextCNN + Glove
| 指标 | 数值 |
|------|:--:|
| Accuracy | **86.78%** |
| Precision (Rumor) | 87.65% |
| Recall (Rumor) | 81.14% |
| F1 Score | 84.27% |
模型: TextCNN + Glove (Embedding dim=200, Filters=(2,3,4,5) × 256, Dropout=0.4, Batch size=32,Learning rate=5e-4)
