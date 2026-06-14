"""
Explanation module — provides LLM-powered natural language explanations
for rumor detection predictions.

Modules:
    - explainer:    Main entry point (RumorExplainer class)
    - llm_client:   LLM API client wrapper
    - prompt_builder: Prompt template construction
"""

from explanation.explainer import RumorExplainer
from explanation.llm_client import LLMClient

__all__ = ["RumorExplainer", "LLMClient"]
