# Interpretability-based Rumor Detection

## 环境要求

- Python 3.11
- PyTorch 2.4.0 (或更高版本兼容版)
- 依赖见下方安装说明

## 安装

```bash
pip install pandas scikit-learn joblib torch

## 项目结构

```
├── run.py                  # BiGRU 训练入口：py -3.11 run.py
├── config/
│   └── config.py           # 超参数与路径配置
├── models/
│   ├── bigru.py            # BiGRU 模型定义
│   └── lr_model.py         # 逻辑回归基线模型 (✨新增加)
├── utils/
│   └── data_utils.py       # 数据加载、分词、词表构建
├── scripts/
│   ├── train.py            # BiGRU 训练 + 评估流程
│   ├── predict.py          # BiGRU 推理模块 (RumorClassifier)
│   ├── train_lr.py         # 逻辑回归 训练 + 评估流程 (✨新增加)
│   └── predict_lr.py       # 逻辑回归 推理模块 (LRRumorClassifier) (✨新增加)
├── data/
│   ├── train.csv           # 训练集
│   └── val.csv             # 验证集
├── checkpoints/
│   ├── bigru.pt            # 训练好的 BiGRU 模型权重
│   ├── vocab.pt            # 词表
│   └── lr_model.pkl        # 训练好的 逻辑回归 模型权重 (✨新增加)
└── interpretability/       # 可解释性模块 (待实现)
```

## 使用方式

### 训练模型

```bash
cd Interpretability-based_rumor_detection
py -3.11 run.py
```

### 推理预测

```python
from scripts.predict import RumorClassifier

clf = RumorClassifier()
result = clf.classify("your text here")   # 0=非谣言, 1=谣言
```

### 训练模型

```bash
cd Interpretability-based_rumor_detection
python scripts/train_lr.py
```

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
