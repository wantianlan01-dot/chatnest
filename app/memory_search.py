"""Lightweight BM25 keyword search for CLAUDE.md - no vector model downloads."""

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(os.environ.get("AGENT_APP_ROOT", Path.cwd())).expanduser().resolve()
MEMORY_FILE = Path(os.environ.get("MEMORY_FILE", ROOT / "CLAUDE.md")).expanduser()
MIN_QUERY_CHARS = 4

_bm25 = None
_corpus = []
_headings = []
_loaded = False


def _load():
    global _bm25, _corpus, _headings, _loaded
    if _loaded:
        return True
    try:
        import jieba
        from rank_bm25 import BM25Okapi

        if not MEMORY_FILE.exists():
            logger.info("No CLAUDE.md found, memory search disabled")
            return False

        text = MEMORY_FILE.read_text(encoding="utf-8")
        chunks = re.split(r"(?=^# )", text, flags=re.MULTILINE)
        _corpus.clear()
        _headings.clear()
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 20:
                continue
            first_line = chunk.split("\n")[0].strip("# ").strip()
            _headings.append(first_line)
            _corpus.append(chunk)

        _bm25 = BM25Okapi([list(jieba.cut(d)) for d in _corpus])
        _loaded = True
        logger.info("Memory search ready: %d sections from CLAUDE.md", len(_corpus))
        return True
    except Exception as e:
        logger.warning("Memory load failed: %s", e)
        return False


def recall(query: str, budget_chars: int = 2000, top_k: int = 8) -> str:
    """BM25 keyword search on CLAUDE.md. Returns joined text or '' on failure."""
    q = (query or "").strip()
    if len(q) < MIN_QUERY_CHARS:
        return ""
    if not _load():
        return ""
    try:
        import jieba

        scores = _bm25.get_scores(list(jieba.cut(q)))
        best = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        seen, chunks = set(), []
        for idx in best:
            if scores[idx] > 0 and _corpus[idx] not in seen:
                seen.add(_corpus[idx])
                chunks.append(f"[{_headings[idx]}]\n{_corpus[idx]}")

        joined = "\n\n".join(chunks)
        if len(joined) > budget_chars:
            joined = joined[:budget_chars] + "..."
        return joined
    except Exception as e:
        logger.warning("Memory search failed: %s", e)
        return ""
