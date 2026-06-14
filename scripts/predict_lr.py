"""
Inference: load trained LR model and predict rumor label for input text(s).
"""
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CHECKPOINT_DIR
from models.lr_model import RumorLR

LR_MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "lr_model.pkl")

class LRRumorClassifier:
    """
    Wrapper for the trained Logistic Regression rumor detection model.
    """
    def __init__(self, model_path: str = None):
        model_path = model_path or LR_MODEL_SAVE_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_lr.py first.")
        
        # 加载本地保存的 pkl 模型数据
        model_data = joblib.load(model_path)
        self.lr_helper = RumorLR()
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']

    def classify(self, text: str) -> int:
        """将单条文本分类为谣言 (1) 或非谣言 (0) """
        cleaned_text = self.lr_helper.preprocess(text)
        X_vec = self.vectorizer.transform([cleaned_text])
        pred = self.model.predict(X_vec)[0]
        return int(pred)

    def classify_batch(self, texts: list) -> list:
        """批量分类文本"""
        return [self.classify(t) for t in texts]

# ================ 交互式本地测试 Demo ================
if __name__ == "__main__":
    if not os.path.exists(LR_MODEL_SAVE_PATH):
        print(f"Error: Model not found at {LR_MODEL_SAVE_PATH}. Please run python scripts/train_lr.py first.")
        sys.exit(1)

    clf = LRRumorClassifier()

    print("=" * 50)
    print("Rumor Detection Demo (Logistic Regression)")
    print("Enter text to classify, or 'q' to quit.")
    print("=" * 50)

    while True:
        try:
            text = input("\nText: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() == "q":
            break
        if not text:
            continue

        pred = clf.classify(text)
        label = "Rumor" if pred == 1 else "Non-rumor"
        print(f"Prediction: {label} ({pred})")