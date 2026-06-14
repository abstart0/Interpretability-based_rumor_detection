"""
LLM API configuration for the explanation module.

Reads API credentials from .env file (via python-dotenv) and exposes
connection parameters for the Shanghai Jiao Tong University LLM API.

Usage:
    from config.llm_config import LLM_API_KEY, LLM_API_BASE, LLM_MODEL_NAME
"""

import os

from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ================ API Connection ================
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://claw.sjtu.edu.cn/api/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4")

# ================ Request Settings ================
# Maximum tokens in the model's response (limit depends on the model)
MAX_TOKENS = 512
# Temperature: 0.0 = deterministic, 1.0 = creative
TEMPERATURE = 0.3
# Timeout in seconds for API requests
TIMEOUT = 60

# ================ Retry Settings ================
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries
