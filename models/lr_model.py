"""
Logistic Regression model for rumor binary classification.
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class RumorLR:
    """
    Logistic Regression Classifier with TF-IDF Vectorizer.
    二分类任务：0代表非谣言，1代表谣言
    """
    def __init__(self, max_features: int = 10000):
        # 初始化文本向量化工具（TF-IDF）和逻辑回归模型 [cite: 34]
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        self.model = LogisticRegression(max_iter=1000)

    def preprocess(self, text: str) -> str:
        """文本预处理：转为小写并移除非字母数字的标点符号 [cite: 34]"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        return re.sub(r'[^\w\s]', '', text)