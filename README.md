# Interpretability-based Rumor Detection

可解释的谣言检测——深度学习模型（TextCNN + GloVe）负责二分类，大语言模型负责输出判断依据。

## 环境要求

- **Python 3.11**
- **PyTorch 2.4.0** (CPU)
- Windows / Linux / macOS

## 安装

```bash
pip install pandas scikit-learn joblib torch==2.4.0
```

> **Windows 注意：**
> - 使用 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 控制 CPU 并行，**不要**调用 `torch.set_num_threads()`
> - DataLoader `num_workers` 必须为 0（spawn 多进程破坏数据完整性）

## 项目结构

```
├── predictor.py              # 推理接口（→ 可解释性模块调用）
├── run_textcnn.py            # 训练入口（TextCNN + GloVe）
├── run_bigru.py              # 训练入口（BiGRU，备用）
├── config/
│   ├── textcnn_config.py     # TextCNN 超参数
│   └── bigru_config.py       # BiGRU 超参数
├── models/
│   ├── textcnn.py            # TextCNN 模型 (Kim 2014)
│   └── bigru.py              # BiGRU 模型
├── utils/
│   ├── data_utils.py         # 分词、词表、Dataset/DataLoader
│   └── glove_utils.py        # GloVe 下载与 Embedding 矩阵构建
├── scripts/
│   ├── train_textcnn.py      # TextCNN 训练脚本
│   ├── train_bigru.py        # BiGRU 训练脚本
│   └── predict_bigru.py      # BiGRU 推理（备用）
├── data/
│   ├── train.csv             # 训练集 2,840 条
│   └── val.csv               # 验证集 401 条
├── checkpoints/              # 训练产物
│   ├── textcnn_glove.pt      # TextCNN 模型权重
│   └── vocab_textcnn.pt      # 词表
├── interpretability/         # 可解释性模块（LLM）
├── notebooks/                # Jupyter notebooks
└── tests/                    # 测试
```

## 快速开始

### 1. 训练深度学习分类模型

首次运行自动下载 GloVe 6B 词向量（~822MB，仅一次）：

```bash
# Windows
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 py -3.11 run_textcnn.py

# Linux / macOS
OMP_NUM_THREADS=10 py -3.11 run_textcnn.py
```

### 2. 推理（供可解释性模块调用）

```python
from predictor import RumorDetector

detector = RumorDetector()
label, confidence = detector.predict("输入文本")
# label: 0（非谣言）或 1（谣言）
# confidence: 模型置信度 (0.0 ~ 1.0)
```

### 3. 可解释性模块

可解释性模块调用 `RumorDetector` 获取分类结果后，结合 LLM 生成判断依据文字。接口约定：

```python
# 输入
text: str          # 推文文本

# 输出
label: int         # 0（非谣言）或 1（谣言）
confidence: float  # 模型置信度，范围 [0, 1]
```

## 模型

| 项目 | 说明 |
|------|------|
| 架构 | TextCNN (Kim 2014) — 多尺寸卷积核 + Max-over-time Pooling |
| 词向量 | GloVe 6B 200d（Wikipedia + Gigaword 预训练，微调），覆盖率 90.9% |
| 输入 | 推文文本（≤64 tokens） |
| 输出 | 二分类 logit → sigmoid → 0/1 + 置信度 |

### 验证集性能

| 指标 | 数值 |
|------|:--:|
| **Accuracy** | **86.78%** |
| Precision (Rumor) | 87.65% |
| Recall (Rumor) | 81.14% |
| F1 Score | 84.27% |

### 超参数

| 参数 | 值 |
|------|-----|
| 词向量维度 | 200 (GloVe 6B) |
| 卷积核 | (2, 3, 4, 5)-gram × 256 |
| Dropout | 0.4 |
| Batch size | 32 |
| Learning rate | 5e-4 (Adam) |
| Weight decay | 2e-4 |
| 词表大小 | 2,727 (min_freq=2) |
| 参数量 | 1,264,249 |
| Epochs | 20 (early stopping patience=7) |
