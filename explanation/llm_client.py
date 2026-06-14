"""
LLM API client for the Shanghai Jiao Tong University AI platform.

Provides a thin wrapper around the OpenAI-compatible Chat Completions API,
with automatic retry on transient failures (network errors, rate limits, etc.).

Usage:
    from explanation.llm_client import LLMClient

    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]
    response = client.chat(messages)
    print(response)  # 'Hello! How can I help you?'
"""

import time
import logging

import requests

from config.llm_config import (
    LLM_API_KEY,
    LLM_API_BASE,
    LLM_MODEL_NAME,
    MAX_TOKENS,
    TEMPERATURE,
    TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    OpenAI-compatible Chat Completions API client.

    Connects to the SJTU AI platform's API endpoint.
    All requests go through the standard chat/completions endpoint.
    """

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        timeout: int = None,
        max_retries: int = None,
        retry_delay: float = None,
    ):
        """
        Args:
            api_key:       Your API key (falls back to env var if None).
            api_base:      API base URL (falls back to config if None).
            model:         Model name (falls back to config if None).
            max_tokens:    Max response tokens (falls back to config if None).
            temperature:   Sampling temperature (falls back to config if None).
            timeout:       Request timeout in seconds (falls back to config if None).
            max_retries:   Max retry attempts on failure (falls back to config if None).
            retry_delay:   Seconds to wait between retries (falls back to config if None).
        """
        self.api_key = api_key or LLM_API_KEY
        self.api_base = (api_base or LLM_API_BASE).rstrip("/")
        self.model = model or LLM_MODEL_NAME
        self.max_tokens = max_tokens or MAX_TOKENS
        self.temperature = temperature if temperature is not None else TEMPERATURE
        self.timeout = timeout or TIMEOUT
        self.max_retries = max_retries or MAX_RETRIES
        self.retry_delay = retry_delay or RETRY_DELAY

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.api_url = f"{self.api_base}/chat/completions"

    def chat(self, messages: list, max_tokens: int = None, temperature: float = None) -> str:
        """
        Send a chat completion request and return the model's text response.

        Args:
            messages:    List of message dicts with "role" and "content".
            max_tokens:  Override default max_tokens for this request.
            temperature: Override default temperature for this request.

        Returns:
            The model's response text (first choice only).

        Raises:
            RuntimeError: If the API returns an error after all retries.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                # Extract text from the first choice
                choice = data["choices"][0]
                return choice["message"]["content"].strip()

            except (requests.RequestException, KeyError, IndexError) as e:
                logger.warning(
                    "API call failed (attempt %d/%d): %s",
                    attempt, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)  # exponential backoff
                else:
                    raise RuntimeError(
                        f"LLM API call failed after {self.max_retries} attempts: {e}"
                    ) from e

        # Should not reach here, but satisfy the type checker
        raise RuntimeError("Unexpected state in LLM API call")
