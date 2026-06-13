# 可解释谣言检测系统

## 项目简介

本项目为《人工智能导论》课程大作业，实现了对社交媒体推文的**谣言二分类检测**，并能**自动生成自然语言的判断依据**。

系统采用“传统检测模型 + 大语言模型解释”的架构：
- 检测模块：实现逻辑回归、RNN等多种基线模型
- 解释模块：调用上海交通大学提供的LLM API，生成可读的判断依据文本

## 小组分工

| 成员   | 任务                                         | 主要代码目录                 |
| ------ | -------------------------------------------- | ---------------------------- |
| 组长   | 项目整合、文档撰写、实验分析、报告汇总       | `/`、`/docs/`                |
| 成员A  | ...模型开发、训练与评估                 | `/baseline_models/.../` |
| 成员B  | ...模型开发、训练与评估                      | `/baseline_models/.../`      |
| 成员C  | LLM API接入、提示词设计、解释模块封装        | `/explanation/`              |

## 环境配置

### 1. 基础环境
- Python 3.11
- Conda（推荐，可选）

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

主要依赖包：
- `torch==2.4.0`
- `pandas`
- `scikit-learn`
- `joblib`
- `requests`（用于API调用）

## 数据集

- `train.csv`：训练集，字段包括 `id`, `text`, `label` (0非谣言/1谣言), `event`
- `val.csv`：验证集，字段相同

请将数据集放在 `data/` 目录下。

## 快速开始(待实验后修改更新)

### 1. 训练检测模型

#### 逻辑回归(例)
```bash
python baseline_models/logistic/train_lr.py
```
训练后模型保存在 `baseline_models/logistic/lr_model.pkl`

#### BiGRU(例)
```bash
python baseline_models/rnn/train_bigru.py
```
训练后模型保存在 `baseline_models/rnn/bigru.pt`

### 2. 配置LLM API

在项目根目录创建 `.env` 文件：
```
LLM_API_KEY=你的交大API密钥
LLM_API_BASE=https://claw.sjtu.edu.cn/api/v1
```
详细API文档参考：[交大AI平台](https://claw.sjtu.edu.cn/guide/sjtu-api/)

### 3. 运行完整检测（含解释）

```bash
python main.py --text "你的推文内容"
```

示例输出：
```json
{
    "prediction": 1,
    "explanation": "该推文使用了'紧急扩散'、'官方尚未证实'等谣言常见表述，且缺乏具体信息来源，因此判定为谣言。"
}
```

### 4. 批量评估验证集
```bash
python evaluate.py --model lr --val-file data/val.csv
```

## 项目结构

```
├── README.md
├── requirements.txt
├── main.py                 # 主入口：整合检测+解释(待写)
├── evaluate.py             # 批量评估脚本(待写)
├── data/                   # 数据集目录
│   ├── train.csv
│   └── val.csv
├── baseline_models/        # 传统检测模型
│   ├── .../
│   │   ├── ...
│   │   ├── ...
│   │   └── ...
│   └── .../
│       ├── ...
│       ├── ...
│       └── ...
├── explanation/            # 可解释性模块
│   ├── ...
│   └── ...
└── docs/                   # 报告相关
    └── report.pdf
```

## 主要结果(暂放虚构例子待更新)

| 模型     | 验证集准确率 | 备注                         |
| -------- | ------------ | ---------------------------- |
| 逻辑回归 | 约 75%       | 快速基线，可解释权重特征     |
| BiGRU    | 约 80-85%    | 捕捉序列信息，性能较优       |
| +LLM解释 | 与检测模型同 | 提供自然语言判断依据，满足要求 |

*（具体数值需根据实际训练更新）*

## 参考资源

- 课程提供的大作业支持文档（逻辑回归与BiGRU参考代码）
- 交大AI平台LLM API文档：https://claw.sjtu.edu.cn/guide/sjtu-api/
```