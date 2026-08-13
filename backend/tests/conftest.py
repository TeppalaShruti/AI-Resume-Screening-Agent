import os
import sys

# Ensure the backend package root is importable and force the offline
# TF-IDF embedding backend so tests are fast and deterministic.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DISABLE_TRANSFORMERS", "1")
os.environ.pop("OPENAI_API_KEY", None)
