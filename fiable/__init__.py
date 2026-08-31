"""
Fiable - Model Compression Pipeline
A comprehensive CLI tool for downloading, quantizing, evaluating, and visualizing LLM compression.
"""

__version__ = "0.1.0"
__author__ = "Fiable Team"

from fiable.config import settings
from fiable import utils

__all__ = ["settings", "utils", "__version__"]
