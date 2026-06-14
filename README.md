# Interpretability-based Rumor Detection

谣言检测 — 二分类（0=非谣言, 1=谣言）。提供 BiGRU 和 TextCNN+GloVe 两种模型。

## 环境要求

- **Python 3.11**
- **PyTorch 2.4.0** (CPU 版)
- Windows / Linux / macOS

## 安装

```bash
pip install pandas scikit-learn joblib torch==2.4.0
```

> **Windows 注意：**
> - 使用 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 控制并行数，**不要**调用 `torch.set_num_threads()`（干扰 OpenMP 调度）
> - DataLoader 的 `num_workers` **必须设为 0**（spawn 多进程破坏数据完整性）

## 项目结构

```
├── run.py                    # BiGRU 训练入口
├── config/
│   ├── config.py             # BiGRU 超参数
│   └── textcnn_config.py     # TextCNN + GloVe 超参数
├── models/
│   ├── bigru.py              # BiGRU 模型
│   └── textcnn.py            # TextCNN 模型 (Kim 2014)
├── utils/
│   ├── data_utils.py         # 分词、词表、Dataset/DataLoader
│   └── glove_utils.py        # GloVe 下载、加载、Embedding 矩阵构建
├── scripts/
│   ├── train.py              # BiGRU 训练
│   ├── train_textcnn.py      # TextCNN + GloVe 训练
│   └── predict.py            # BiGRU 推理 (RumorClassifier)
├── data/
│   ├── train.csv             # 训练集 2,840 条
│   └── val.csv               # 验证集 401 条
├── checkpoints/
│   ├── bigru.pt / textcnn_glove.pt   # 模型权重
│   └── vocab*.pt                     # 词表
└── interpretability/         # 可解释性模块 (待实现)
```

## 使用方式

### BiGRU 训练

```bash
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 py -3.11 run.py
```

### TextCNN + GloVe 训练

首次运行会自动下载 GloVe 6B 词向量（~822MB，仅一次）：

```bash
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 py -3.11 scripts/train_textcnn.py
```

### BiGRU 推理

```python
from scripts.predict import RumorClassifier

clf = RumorClassifier()
result = clf.classify("Breaking: scientists discover alien life!")  # → 0 or 1
```

## 模型性能

### 完整对比

| 模型 | Acc | Prec | Rec | F1 | Params | 关键改进 |
|------|:--:|:--:|:--:|:--:|:--:|------|
| BiGRU 基线 | 85.04% | 93.23% | 70.86% | 80.52% | 449K | 支持文档参考代码 |
| BiGRU 最优 | 85.29% | 86.71% | 78.29% | 82.28% | 832K | 网格搜索 + 去 embed_dropout |
| TextCNN (无预训练) | 86.03% | — | — | — | 678K | 模型结构优势 |
| TextCNN+GloVe v1 | 86.28% | 89.47% | 77.71% | 83.18% | 1.82M | +GloVe, min_freq=1 |
| **TextCNN+GloVe v2** | **86.78%** | **87.65%** | **81.14%** | **84.27%** | **1.26M** | **+GloVe + min_freq=2 + 256f** |

### TextCNN+GloVe v2 — 最终最优

| 参数 | 值 |
|------|-----|
| 模型 | TextCNN (Kim 2014) |
| 词向量 | GloVe 6B 200d（预训练 + 微调），覆盖率 90.9% |
| 卷积核 | (2,3,4,5)-gram × 256 |
| Dropout | 0.4 |
| Batch size | 32 |
| Learning rate | 5e-4 (Adam) |
| Weight decay | 2e-4 |
| 词表 | 2,727 (min_freq=2) |
| 参数量 | 1,264,249 |

### BiGRU 最优

| 参数 | 值 |
|------|-----|
| Embedding dim | 150 |
| Hidden dim | 200 |
| Layers | 1 |
| Dropout | 0.5 |
| Weight decay | 3e-4 |
| Batch size | 32 |
| Learning rate | 5e-4 (Adam) |
| 参数量 | 831,851 |
