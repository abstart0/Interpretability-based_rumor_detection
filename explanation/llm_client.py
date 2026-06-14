"""
LLM API 客户端封装

负责与上海交通大学LLM平台通信，包含请求构造、重试机制、超时控制。
"""

import os
import time
import requests
from dotenv import load_dotenv

# 自动加载 .env 文件中的环境变量
load_dotenv()

DEFAULT_API_BASE = os.getenv("LLM_API_BASE", "https://claw.sjtu.edu.cn/api/v1")
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3


class LLMClient:
    """
    交大LLM API客户端

    用法:
        client = LLMClient()
        response = client.chat("你的问题")
    """

    def __init__(self, api_base: str = None, api_key: str = None, model: str = None):
        self.api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
        self.api_key = api_key or DEFAULT_API_KEY
        self.model = model or DEFAULT_MODEL

        if not self.api_key:
            raise ValueError(
                "未检测到LLM API密钥。\n"
                "请在项目根目录创建 .env 文件，添加:\n"
                "  LLM_API_KEY=你的密钥\n"
                "  LLM_API_BASE=https://claw.sjtu.edu.cn/api/v1"
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: list, temperature: float = 0.3, max_tokens: int = 500) -> str:
        """
        发送聊天请求到LLM API。

        Args:
            messages: 消息列表，格式 [{"role": "user/system", "content": "..."}]
            temperature: 采样温度，越低越确定
            max_tokens: 最大生成长度

        Returns:
            LLM返回的文本内容

        Raises:
            RuntimeError: API调用失败或超时
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        url = f"{self.api_base}/chat/completions"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(url, headers=self.headers, json=payload, timeout=TIMEOUT_SECONDS)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout:
                print(f"  [警告] API调用超时 (尝试 {attempt}/{MAX_RETRIES})")
            except requests.exceptions.HTTPError as e:
                # 4xx 错误通常不需要重试（如密钥错误）
                if resp.status_code < 500:
                    raise RuntimeError(f"LLM API 请求失败 (HTTP {resp.status_code}): {resp.text}") from e
                print(f"  [警告] API服务器错误 {resp.status_code} (尝试 {attempt}/{MAX_RETRIES})")
            except Exception as e:
                print(f"  [警告] 未知错误: {e} (尝试 {attempt}/{MAX_RETRIES})")

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  # 指数退避

        raise RuntimeError(f"LLM API 调用失败，已重试 {MAX_RETRIES} 次")
