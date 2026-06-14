"""
主入口脚本

用法:
    python main.py --text "你的推文内容"

流程:
    1. 调用 RumorDetector (TextCNN+GloVe) 进行谣言检测
    2. 将检测结果传递给 Explainer
    3. 调用交大 LLM API 生成自然语言解释
    4. 输出 JSON 格式结果
"""

import argparse
import json
import os
import sys

# Windows OpenMP 冲突修复
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predictor import RumorDetector
from scripts.predict_lr import LRRumorClassifier
from scripts.predict_bigru import RumorClassifier
from explanation.explainer import Explainer


def _get_detector(model_name: str):
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
    parser = argparse.ArgumentParser(description="可解释谣言检测系统")
    parser.add_argument("--text", type=str, required=True, help="待检测的推文内容")
    parser.add_argument("--model", type=str, default="textcnn",
                        choices=["textcnn", "bigru", "lr"],
                        help="使用的检测模型 (默认: textcnn)")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 格式输出结果")
    args = parser.parse_args()

    # Step 1: 加载检测模型
    print(f"正在加载检测模型 ({args.model})...")
    detector = _get_detector(args.model)

    # Step 2: 执行检测
    if hasattr(detector, "predict"):
        label, confidence = detector.predict(args.text)
    else:
        label = detector.classify(args.text)
        confidence = 0.5  # LR/BiGRU 独立推理接口不提供置信度
    label_desc = "谣言" if label == 1 else "非谣言"
    print(f"检测结果: {label_desc} (置信度: {confidence:.1%})")

    # Step 3: 生成解释
    print("正在生成解释...")
    try:
        explainer = Explainer()
        explanation = explainer.explain(args.text, label, confidence)
        print(f"解释: {explanation}")
    except Exception as e:
        explanation = f"解释生成失败: {e}"
        print(f"[错误] {explanation}")

    # Step 4: 输出结果（按大作业要求的两个输出来展示）
    result = {
        "text": args.text,
        "prediction": label,
        "label": label_desc,
        "confidence": round(confidence, 4),
        "explanation": explanation,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=4))
    else:
        print("\n========================================")
        print("         可解释谣言检测结果")
        print("========================================")
        print(f"输入文本: {args.text}")
        print(f"检测输出: {label_desc} (类别 {label}, 置信度 {confidence:.1%})")
        print(f"判断依据: {explanation}")
        print("========================================")


if __name__ == "__main__":
    main()
