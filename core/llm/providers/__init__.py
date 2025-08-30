"""
core.llm.providers
=================

Provider-specific LLM implementations.

This package contains implementations of the BaseLLM interface for different providers.
"""

# Import specific providers so they register themselves
from core.llm.providers.openai import OpenAILLM

__all__ = [
    "OpenAILLM"
]
