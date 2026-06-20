"""
Core explanation generator: bridges the rumor detection model and the LLM.

This is the main entry point for the explainability component. It:
  1. Loads the detection model (TextCNN by default, supports lr/bigru)
  2. Runs prediction on the input text
  3. Constructs a prompt with the prediction result
  4. Calls the LLM to generate a natural language explanation

Usage:
    from explanation.explainer import RumorExplainer

    explainer = RumorExplainer(model="textcnn")
    result = explainer.explain("Breaking news: huge explosion downtown!")
    # result = {"label": 1, "confidence": 0.92, "explanation": "该推文使用了..."}
"""

import logging

from predictor import RumorDetector
from explanation.llm_client import LLMClient
from explanation.prompt_builder import build_explanation_prompt

logger = logging.getLogger(__name__)


class RumorExplainer:
    """
    End-to-end rumor detector + explanation generator.

    Combines a rumor detection model with an LLM to produce
    natural language explanations for each prediction.

    Supported models: textcnn (default), bigru, lr
    """

    def __init__(self, model: str = "textcnn",
                 model_path: str = None, vocab_path: str = None,
                 llm_client: LLMClient = None):
        """
        Args:
            model:        Detection model name: "textcnn", "bigru", or "lr".
            model_path:   Path to the detection model checkpoint.
            vocab_path:   Path to the vocabulary file.
            llm_client:   Pre-configured LLMClient (creates a default one if None).
        """
        self.model_name = model
        if model == "textcnn":
            self.detector = RumorDetector(model_path=model_path, vocab_path=vocab_path)
        elif model == "bigru":
            from scripts.predict_bigru import RumorClassifier
            self._bigru_clf = RumorClassifier(
                model_path=model_path, vocab_path=vocab_path
            )
            self.detector = None
        elif model == "lr":
            from scripts.predict_lr import LRRumorClassifier
            self._lr_clf = LRRumorClassifier(model_path=model_path)
            self.detector = None
        else:
            raise ValueError(f"Unsupported model: {model}. Choose from: textcnn, bigru, lr")

        self.llm_client = llm_client or LLMClient()

    def explain(self, text: str, llm_model: str = None,
                max_tokens: int = None, temperature: float = None) -> dict:
        """
        Predict and generate an explanation for the input text.

        Args:
            text:          The input tweet or text to analyze.
            llm_model:     Override the LLM model name for this call.
            max_tokens:    Override max tokens for the LLM response.
            temperature:   Override temperature for the LLM response.

        Returns:
            dict with keys:
                - label (int): 0=non-rumor, 1=rumor
                - confidence (float): model confidence in [0, 1]
                - explanation (str): natural language explanation from LLM

        Raises:
            RuntimeError: If the LLM call fails after retries.
        """
        # Step 1: Run detection model
        if self.model_name == "textcnn":
            label, confidence = self.detector.predict(text)
        elif self.model_name == "bigru":
            label = self._bigru_clf.classify(text)
            confidence = 0.5  # BiGRU does not expose confidence
        elif self.model_name == "lr":
            label = self._lr_clf.classify(text)
            confidence = 0.5  # LR does not expose confidence

        # Step 2: Build prompt with prediction result
        messages = build_explanation_prompt(text, label, confidence)

        # Step 3: Get explanation from LLM
        explanation = self.llm_client.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "label": int(label),
            "confidence": float(confidence),
            "explanation": explanation,
        }

    def explain_batch(self, texts: list, **kwargs) -> list:
        """
        Explain a batch of texts sequentially.

        Args:
            texts:   List of input text strings.
            **kwargs: Forwarded to self.explain() (e.g., llm_model, temperature).

        Returns:
            List of result dicts, same format as explain().
        """
        return [self.explain(t, **kwargs) for t in texts]
