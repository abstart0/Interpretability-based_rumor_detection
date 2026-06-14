"""
批量评估脚本

用法:
    python evaluate.py --model lr --val-file data/val.csv

对验证集中每条推文依次执行检测+LLM解释，统计检测结果分布，
并将完整结果保存到 JSON 文件。
"""

import argparse
import json
import os
import sys
import time

# Windows OpenMP 冲突修复
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

# 导入各模型的推理类
from predictor import RumorDetector
from scripts.predict_lr import LRRumorClassifier
from scripts.predict_bigru import RumorClassifier
from explanation.explainer import Explainer


def get_detector(model_name: str):
    """根据模型名称返回对应的检测器实例"""
    if model_name == "textcnn":
        return RumorDetector()
    elif model_name == "lr":
        return LRRumorClassifier()
    elif model_name == "bigru":
        return RumorClassifier()
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="批量评估谣言检测+解释")
    parser.add_argument("--model", type=str, default="textcnn",
                        choices=["textcnn", "bigru", "lr"],
                        help="使用的检测模型 (默认: textcnn)")
    parser.add_argument("--val-file", type=str, default="data/val.csv",
                        help="验证集 CSV 路径")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="结果输出 JSON 路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅检测，不调用 LLM 生成解释（节省 API 调用）")
    args = parser.parse_args()

    # 加载验证集
    print(f"加载验证集: {args.val_file}")
    df = pd.read_csv(args.val_file)
    print(f"共 {len(df)} 条数据")

    # 初始化检测器和解释器
    print(f"初始化检测模型: {args.model}")
    detector = get_detector(args.model)

    explainer = None
    if not args.dry_run:
        print("初始化 LLM 解释器...")
        try:
            explainer = Explainer()
        except ValueError as e:
            print(f"[警告] LLM 未配置 ({e})，跳过解释生成，使用 --dry-run 模式")
            args.dry_run = True

    # 逐条评估
    results = []
    true_pos = false_pos = true_neg = false_neg = 0

    print("\n开始评估...")
    for i, row in df.iterrows():
        text = row["text"]
        true_label = int(row["label"])

        # 检测
        if hasattr(detector, "predict"):
            pred, confidence = detector.predict(text)
        else:
            pred = detector.classify(text)
            confidence = 0.5  # LR 不提供置信度

        # 混淆矩阵统计
        if pred == 1 and true_label == 1:
            true_pos += 1
        elif pred == 1 and true_label == 0:
            false_pos += 1
        elif pred == 0 and true_label == 0:
            true_neg += 1
        else:
            false_neg += 1

        # 解释
        explanation = None
        if explainer is not None:
            try:
                explanation = explainer.explain(text, pred, confidence)
            except Exception as e:
                explanation = f"解释失败: {e}"

        results.append({
            "id": row.get("id", i),
            "text": text,
            "true_label": true_label,
            "pred_label": pred,
            "confidence": round(float(confidence), 4),
            "correct": pred == true_label,
            "explanation": explanation,
        })

        label_str = "谣言" if pred == 1 else "非谣言"
        status = "正确" if pred == true_label else "错误"
        print(f"  [{i+1}/{len(df)}] {label_str} (置信度 {confidence:.2f}) "
              f"| 真实: {'谣言' if true_label == 1 else '非谣言'} | {status}")

        # 避免 API 限流，简单延迟
        if not args.dry_run and i < len(df) - 1:
            time.sleep(0.5)

    # 统计指标
    total = len(df)
    correct = sum(1 for r in results if r["correct"])
    acc = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"评估完成 ({args.model})")
    print(f"{'='*50}")
    print(f"准确率: {acc:.4f} ({correct}/{total})")
    print(f"混淆矩阵: TP={true_pos} FP={false_pos} TN={true_neg} FN={false_neg}")

    # 保存结果
    output_data = {
        "model": args.model,
        "total": total,
        "accuracy": round(acc, 4),
        "confusion_matrix": {
            "TP": true_pos, "FP": false_pos,
            "TN": true_neg, "FN": false_neg,
        },
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
