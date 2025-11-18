'''Formatting helpers for conversational outputs.'''
from __future__ import annotations
import math
import re
from typing import Iterable, Sequence
import pandas as pd

def normalize_rag_text(text = None):
    '''Reduce excessive newlines and tidy up spacing in RAG answers.'''
    if not text:
        return ''
    cleaned = re.sub('\\n{3,}', '\n\n', text)
    cleaned = re.sub('[\\t\\x0b\\x0c\\r ]+\\n', '\n', cleaned)
    return cleaned.strip()


def format_sql_result_as_markdown(rows, columns):
    '''Return a well-aligned Markdown table for the SQL output, only if >1 rows or >2 columns.'''
    if not rows:
        return None
    if len(rows) == 1 and len(columns) <= 2:
        row_str = ', '.join(f'{columns[i]}: {rows[0][i]}' for i in range(len(columns)))
        return f'**Result:** {row_str}'
    try:
        import pandas as pd
        df = pd.DataFrame(rows, columns=columns)
        return df.to_markdown(index=False)
    except ImportError:
        return '\n'.join(', '.join(map(str, r)) for r in rows[:5])

__all__ = [
    'normalize_rag_text',
    'format_sql_result_as_markdown']
