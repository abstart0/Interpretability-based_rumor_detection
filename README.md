# Interpretability-based Rumor Detection

谣言检测深度学习模型 — 二分类（0=非谣言, 1=谣言）。

## 环境要求

- **Python 3.11**
- **PyTorch 2.4.0** (CPU)

## 安装

```bash
pip install pandas scikit-learn joblib torch==2.4.0
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
│   ├── textcnn.py           # TextCNN 模型 (Kim 2014)
│   └── bigru.py             # BiGRU 模型
├── utils/
│   ├── data_utils.py        # 分词、词表、Dataset/DataLoader
│   └── glove_utils.py       # GloVe 下载与 embedding 矩阵构建
├── scripts/
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

## 模型

| 项目 | 说明 |
|------|------|
| 架构 | TextCNN (Kim 2014) — 多尺寸卷积核 + Max-over-time Pooling |
| 词向量 | GloVe 6B 200d（Wikipedia + Gigaword 预训练），覆盖率 90.9% |
| 卷积核 | 2/3/4/5-gram × 256 filters |
| 参数量 | 1,264,249 |

### 验证集性能

| 指标 | 数值 |
|------|:--:|
| Accuracy | **86.78%** |
| Precision (Rumor) | 87.65% |
| Recall (Rumor) | 81.14% |
| F1 Score | 84.27% |

### 超参数

| 参数 | 值 |
|------|-----|
| Embedding dim | 200 (GloVe) |
| Filters | (2, 3, 4, 5) × 256 |
| Dropout | 0.4 |
| Batch size | 32 |
| Learning rate | 5e-4 (Adam) |
| Weight decay | 2e-4 |
| 词表（min_freq=2） | 2,727 |
| Epochs | 20 (early stopping patience=7) |
