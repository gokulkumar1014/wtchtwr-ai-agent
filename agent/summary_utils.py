"""Shared helpers for adaptive summarisation across SQL and RAG pipelines."""
from __future__ import annotations
import re
_STOPWORDS = {'the', 'and', 'on', 'list', 'what', 'with', 'for', 'me', 'show', 'give', 'reviews', 'is', 'based', 'tell'}

def extract_focus_word(query: str) -> str:
    """Pull the most relevant non-stopword token from the user query."""
    tokens = re.findall('[a-zA-Z]+', (query or '').lower())
    for token in reversed(tokens):
        if token and token not in _STOPWORDS:
            return token
    else:
        return 'insight'
__all__ = ['extract_focus_word']