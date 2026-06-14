"""
Prompt template builder for generating natural language explanations.

Constructs carefully structured prompts that combine:
  1. The detected rumor label and confidence score from the detection model
  2. The original input text
  3. A role-playing system prompt that guides the LLM to produce
     professional, evidence-based explanation text.

Each prompt follows a consistent structure:
  [system] Role definition + output format requirements
  [user]  Input text + model prediction result + request for explanation
"""

SYSTEM_PROMPT = (
    "你是一位专业的社交媒体信息审核员。你的任务是根据提供的推文内容和谣言检测模型的判断结果，"
    "生成一段判断依据的文字，说明为什么该推文被判定为谣言或非谣言。"
    "\n\n"
    "要求：\n"
    "1. 解释必须基于文本本身的语言特征和事实线索\n"
    "2. 指出具体的关键词、句式或信息来源特征\n"
    "3. 语气客观、专业，避免主观臆断\n"
    "4. 长度控制在 50-100 个中文字符\n"
    "5. 不要提及模型、置信度、technical术语\n"
    "6. 只输出解释文本，不要添加标题或其他内容\n"
    "7. 重点说明判断的依据，即解释为什么这样判断"
)


def build_explanation_prompt(text: str, label: int, confidence: float) -> list:
    """
    Build the messages list for the LLM API call.

    Args:
        text:       The input tweet/text to be explained.
        label:      0 = non-rumor, 1 = rumor (from detection model).
        confidence: Model confidence in [0.0, 1.0].

    Returns:
        list of message dicts: [
            {"role": "system", "content": ...},
            {"role": "user", "content": ...}
        ]
    """
    label_text = "谣言" if label == 1 else "非谣言"
    # Build a user-facing summary that the LLM can reason from
    user_content = (
        f"请根据以下信息，生成一段对该推文的判断解释。\n\n"
        f"---\n"
        f"推文内容：{text}\n"
        f"检测判定：{label_text}（模型置信度：{confidence:.1%}）\n"
        f"---\n"
        f"你的解释："
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
