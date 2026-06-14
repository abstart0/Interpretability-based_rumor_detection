# Interpretability-based Rumor Detection

BiGRU 谣言检测模型 — 二分类（0=非谣言, 1=谣言）。

## 环境要求

- **Python 3.11**
- **PyTorch 2.4.0** (CPU 版)
- Windows / Linux / macOS

## 安装

```bash
pip install pandas scikit-learn joblib torch==2.4.0
```

> **Windows 注意：** 训练时使用 `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 环境变量控制并行数，**不要**在代码中调用 `torch.set_num_threads()`（会干扰 OpenMP 调度导致模型退化）。

## 项目结构

```
├── run.py                  # 训练入口
├── config/
│   └── config.py           # 超参数与路径
├── models/
│   └── bigru.py            # BiGRU 模型 (Embedding → BiGRU → Dropout → Linear)
├── utils/
│   └── data_utils.py       # 分词、词表构建、Dataset/DataLoader
├── scripts/
│   ├── train.py            # 训练 (Adam + ReduceLROnPlateau + Early Stopping)
│   └── predict.py          # RumorClassifier 推理类
├── data/
│   ├── train.csv           # 训练集 (2,840 条)
│   └── val.csv             # 验证集 (401 条)
├── checkpoints/
│   ├── bigru.pt            # 模型权重
│   └── vocab.pt            # 词表
├── interpretability/       # 可解释性模块 (待实现)
├── notebooks/              # Jupyter notebooks (待实现)
└── tests/                  # 单元测试 (待实现)
```

## 使用方式

### 训练

```bash
cd Interpretability-based_rumor_detection

# Windows (10核示例)
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 py -3.11 run.py

# Linux / macOS
OMP_NUM_THREADS=10 py -3.11 run.py
```

### 推理

```python
from scripts.predict import RumorClassifier

clf = RumorClassifier()
result = clf.classify("Breaking: scientists discover alien life!")  # → 0 or 1
```

## 模型性能

| 指标 | 数值 |
|------|------|
| **Accuracy** | **85.29%** |
| Precision (Rumor) | 86.71% |
| Recall (Rumor) | 78.29% |
| F1 Score | 82.28% |

### 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Embedding dim | 150 | 词向量维度 |
| Hidden dim | 200 | GRU 隐状态维度 |
| Layers | 1 | 单层 BiGRU（多层过拟合） |
| Dropout | 0.5 | GRU 输出后 |
| Batch size | 32 | 小 batch 梯度估计更准 |
| Learning rate | 5e-4 | Adam |
| Weight decay | 3e-4 | L2 正则 |
| Vocabulary | 2,727 | min_freq=2 |
| 参数量 | 831,851 | — |

### 优化迭代历史

| 版本 | 核心改动 | Accuracy | F1 |
|------|---------|:--:|:--:|
| 基线 | 支持文档参考参数 (100/128/dr0.3) | 85.04% | 80.52% |
| 最终 | 网格搜索最优参数 (150/200/dr0.5/wd3e-4) | **85.29%** | **82.28%** |
