"""
解释器核心模块

接收检测模型的输出（文本、标签、置信度），拼接 prompt 调用 LLM，
生成自然语言形式的判断依据。
"""

from explanation.llm_client import LLMClient


# ================ 提示词设计 ================

_SYSTEM_PROMPT = """\
你是一个社交媒体谣言检测专家助手。你的任务是根据自动化检测模型的分析结果，\
为用户提供清晰、客观、易懂的自然语言解释。\
如果检测为谣言，请指出推文中的谣言特征（如夸张措辞、缺乏来源、煽动性语气等）；\
如果检测为非谣言，请说明推文的合理之处（如表述客观、有具体事实、无煽动性词汇等）。
"""

_USER_PROMPT_TEMPLATE = """\
请根据以下检测结果，用中文写一段简短的谣言判断理由（30-80字）。

原始推文: {text}
检测结论: {label_desc}
模型置信度: {confidence:.1%}

请结合推文内容给出理由：
- 如果是谣言，指出其中的可疑特征（如夸张措辞、缺乏权威来源、煽动性语气等）
- 如果是非谣言，说明其合理之处（如表述客观、内容具体、无煽动性词汇等）
- 语气客观，不要提及"模型"、"置信度"等技术术语
- 只输出理由文本，不要输出其他内容
"""


def _format_label_desc(label: int, confidence: float) -> str:
    """将检测结果格式化为人类可读描述"""
    if label == 1:
        return f"谣言 (置信度 {confidence:.1%})"
    return f"非谣言 (置信度 {confidence:.1%})"


class Explainer:
    """
    可解释性解释器

    用法:
        explainer = Explainer()
        explanation = explainer.explain("推文内容", label=1, confidence=0.85)
    """

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()

    def explain(self, text: str, label: int, confidence: float) -> str:
        """
        根据检测结果生成自然语言解释。

        Args:
            text: 原始推文文本
            label: 预测标签 (0=非谣言, 1=谣言)
            confidence: 模型置信度 [0, 1]

        Returns:
            自然语言解释文本
        """
        label_desc = _format_label_desc(label, confidence)

        user_msg = _USER_PROMPT_TEMPLATE.format(
            label_desc=label_desc,
            confidence=confidence,
            text=text,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        return self.llm.chat(messages)
