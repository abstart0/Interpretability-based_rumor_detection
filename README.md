# Interpretability-based Rumor Detection

## 环境要求

- Python 3.11
- PyTorch 2.4.0 (CPU)
- 依赖见下方安装说明

## 安装

```bash
pip install pandas scikit-learn joblib torch==2.4.0
```

## 项目结构

```
├── run.py                  # 训练入口：py -3.11 run.py
├── config/
│   └── config.py           # 超参数与路径配置
├── models/
│   └── bigru.py            # BiGRU 模型定义
├── utils/
│   └── data_utils.py       # 数据加载、分词、词表构建
├── scripts/
│   ├── train.py            # 训练 + 评估流程
│   └── predict.py          # 推理模块 (RumorClassifier)
├── data/
│   ├── train.csv           # 训练集
│   └── val.csv             # 验证集
├── checkpoints/
│   ├── bigru.pt            # 训练好的模型权重
│   └── vocab.pt            # 词表
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

## 模型性能

| 指标 | 数值 |
|------|------|
| Accuracy | 85.04% |
| Precision | 93.23% |
| Recall | 70.86% |
| F1 Score | 80.52% |

模型: BiGRU (embedding=100, hidden=128, vocab=2727, params=449,597)