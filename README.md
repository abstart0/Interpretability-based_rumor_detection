# 可解释谣言检测系统

## 项目简介

本项目为《人工智能导论》课程大作业，实现了对社交媒体推文的**谣言二分类检测**，并能**自动生成自然语言的判断依据**。

系统采用"传统检测模型 + 大语言模型解释"的复合架构：
- 检测模块：实现逻辑回归、RNN 等多种基线模型
- 解释模块：调用上海交通大学"致远一号"LLM API，生成可读的判断依据文本

对应大作业要求：
- **输出1**：检测输出为 2 分类（0 非谣言、1 谣言），分类准确率越高越好
- **输出2**：判断依据为一段中文文字，解释判断的理由

## 小组分工

| 成员   | 任务                                         | 主要代码目录                  |
| ------ | -------------------------------------------- | ----------------------------- |
| 组长   | 项目整合、文档撰写、实验分析、报告汇总         | `/`, `/docs/`                 |
| 成员A  | 模型开发、训练与评估                           | `/baseline_models/.../`       |
| 成员B  | 模型开发、训练与评估                           | `/baseline_models/.../`       |
| 成员C  | LLM API 接入、提示词设计、解释模块封装         | `/explanation/`               |

## 环境配置

### 1. 基础环境

- Python 3.11
- PyTorch 2.4.0（或更高兼容版本）
- Windows / Linux / macOS

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- `torch==2.4.0`
- `pandas`
- `scikit-learn`
- `joblib`
- `requests`（用于 LLM API 调用）
- `python-dotenv`（环境变量配置）

## 数据集

- `data/train.csv`：训练集，字段包括 `id`, `text`, `label` (0非谣言/1谣言), `event`
- `data/val.csv`：验证集，字段相同

## 项目结构

```
├── main.py                 # 统一推理入口（供大作业演示调用）
├── evaluate.py             # 批量评估脚本（检测+解释）
├── run_bigru.py            # 训练入口（BiGRU）
├── run_textcnn.py          # 训练入口（TextCNN + GloVe）
├── config/
│   ├── config.py           # 公共配置（供 LR 模型使用）
│   ├── textcnn_config.py   # TextCNN 超参数
│   └── bigru_config.py     # BiGRU 超参数
├── models/
│   ├── lr_model.py         # 逻辑回归模型
│   ├── bigru.py            # BiGRU 模型
│   └── textcnn.py          # TextCNN 模型 (Kim 2014)
├── utils/
│   ├── data_utils.py       # 分词、词表、Dataset/DataLoader
│   └── glove_utils.py      # GloVe 下载与 Embedding 矩阵构建
├── scripts/
│   ├── train_lr.py         # 逻辑回归训练
│   ├── predict_lr.py       # 逻辑回归推理
│   ├── train_bigru.py      # BiGRU 训练
│   ├── train_textcnn.py    # TextCNN + GloVe 训练
│   └── predict_bigru.py    # BiGRU 推理
├── explanation/            # 可解释性模块（成员C负责）
│   ├── __init__.py         # 包初始化，暴露 Explainer 类
│   ├── llm_client.py       # 交大 LLM API 客户端封装
│   └── explainer.py        # 解释器：拼接 prompt + 调用 LLM + 返回解释
├── predictor.py            # 统一推理接口（RumorDetector）
├── data/
│   ├── train.csv           # 训练集 2,840 条
│   └── val.csv             # 验证集 401 条
├── checkpoints/            # 训练产物
│   ├── lr_model.pkl        # 逻辑回归模型 + 向量器
│   ├── bigru.pt / vocab.pt # BiGRU 权重 + 词表
│   └── textcnn_glove.pt    # TextCNN 权重 + 词表
└── .env                    # LLM API 密钥配置（需用户自行创建）
```

## 快速开始

### 1. 训练检测模型

```bash
# 逻辑回归（最快，~10 秒）
python scripts/train_lr.py

# BiGRU（~2 分钟）
python run_bigru.py

# TextCNN + GloVe（最优，~5 分钟，首次下载 GloVe ~822MB）
python run_textcnn.py
```

训练后模型保存在 `checkpoints` 文件夹下。

### 2. 配置 LLM API

在项目根目录创建 `.env` 文件：

```
LLM_API_KEY=你的交大API密钥
LLM_API_BASE=https://models.sjtu.edu.cn/api/v1
LLM_MODEL=deepseek-reasoner
```

> LLM_MODEL 可选值：`deepseek-chat`、`deepseek-reasoner`、`qwen`、`glm`、`minimax` 等（交大平台支持的模型均可）。
> 使用 `deepseek-chat` 响应更快，使用 `deepseek-reasoner` 会输出思考过程。

### 3. 运行完整检测（含解释）

```bash
# 默认使用 TextCNN + GloVe 模型
python main.py --text "你的推文内容"
```

示例输出：

```
正在加载检测模型 (textcnn)...
检测结果: 谣言 (置信度: 67.9%)
正在生成解释...
解释: 该推文使用"BREAKING"等夸张措辞，缺乏具体科学证据和权威来源...

========================================
         可解释谣言检测结果
========================================
输入文本: Scientists confirm COVID-19 was artificially created in a Chinese lab
检测输出: 谣言 (类别 1, 置信度 67.9%)
判断依据: 该推文使用"BREAKING"等夸张措辞，缺乏具体科学证据和权威来源，且"artificially created"带有强烈暗示性，易引发误导。
========================================
```

支持切换检测模型：

```bash
python main.py --text "your tweet" --model lr
python main.py --text "your tweet" --model bigru
```

JSON 格式输出（便于程序解析）：

```bash
python main.py --text "your tweet" --json
```

```json
{
    "text": "Scientists confirm...",
    "prediction": 1,
    "label": "谣言",
    "confidence": 0.6786,
    "explanation": "该推文使用 BREAKING 等夸张措辞..."
}
```

### 4. 批量评估验证集

```bash
# 仅检测，不调用 LLM（节省 API）
python evaluate.py --model textcnn --dry-run

# 完整评估（检测 + LLM 解释）
python evaluate.py --model textcnn --output val_results.json
```



## 模型性能

### TextCNN + GloVe（最优）

| 指标 | 数值 |
|------|:--:|
| Accuracy | **86.78%** |
| Precision | 87.65% |
| Recall | 81.14% |
| F1 Score | 84.27% |

| 超参数 | 值 |
|------|-----|
| 词向量 | GloVe 6B 200d（预训练 + 微调），覆盖率 90.9% |
| 卷积核 | (2, 3, 4, 5)-gram × 256 |
| Dropout | 0.4 |
| Batch size | 32 |
| Learning rate | 5e-4 (Adam) |
| Weight decay | 2e-4 |
| 词表 | 2,727 (min_freq=2) |
| 参数量 | 1,264,249 |

### BiGRU

| 指标 | 数值 |
|------|:--:|
| Accuracy | 85.29% |
| Precision | 86.71% |
| Recall | 78.29% |
| F1 Score | 82.28% |

| 超参数 | 值 |
|------|-----|
| Embedding dim | 150 |
| Hidden dim | 200 |
| Dropout | 0.5 |
| Batch size | 32 |
| Learning rate | 5e-4 (Adam) |
| Weight decay | 3e-4 |
| 参数量 | 831,851 |

### 逻辑回归

| 指标 | 数值 |
|------|:--:|
| Accuracy | 82.29% |
| Precision | 84.67% |
| Recall | 72.57% |
| F1 Score | 78.15% |

| 超参数 | 值 |
|------|-----|
| 特征 | TF-IDF（max_features=10,000） |
| 模型 | LogisticRegression（max_iter=1,000） |
| 参数量 | 10,001 |



## 参考资源

- 交大 AI 平台 LLM API 文档：https://claw.sjtu.edu.cn/guide/sjtu-api/
- TextCNN 论文：Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification
