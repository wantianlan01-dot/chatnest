"""Local memory search using ChromaDB + BM25, vectorizes on first use."""

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(os.environ.get("AGENT_APP_ROOT", Path.cwd())).expanduser().resolve()
MEMORY_FILE = Path(os.environ.get("MEMORY_FILE", ROOT / "CLAUDE.md")).expanduser()
PROFILE_FILE = Path(os.environ.get("PROFILE_FILE", ROOT / "profile.json")).expanduser()
MEMORIES_DIR = Path(os.environ.get("MEMORIES_DIR", ROOT / "memories")).expanduser()
DB_DIR = Path(os.environ.get("MEMORY_CHROMA_DIR", ROOT / "memory" / "chroma_db")).expanduser()
STATE_FILE = Path(os.environ.get("MEMORY_STATE_FILE", ROOT / "memory" / "chroma_state.json")).expanduser()
COLLECTION = os.environ.get("MEMORY_SEARCH_COLLECTION", "memories")
MIN_QUERY_CHARS = 4

_ready = False
_client = None
_coll = None
_bm25_index = None
_bm25_corpus = []
_vec_lock = threading.Lock()


def _ensure():
    """Lazy init: vectorize on first call. Thread-safe."""
    global _ready, _client, _coll, _bm25_index, _bm25_corpus
    if _ready:
        return True
    with _vec_lock:
        if _ready:
            return True
        try:
            import chromadb
            from memory_search_service.vectorize import (
                collect_sources, source_hash, split_text, delete_sources, load_state, save_state,
            )

            DB_DIR.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(path=str(DB_DIR))
            _coll = _client.get_or_create_collection(COLLECTION)

            state = load_state()
            docs = collect_sources()
            new_state = {doc.source: source_hash(doc) for doc in docs}

            removed = sorted(set(state) - set(new_state))
            if removed:
                delete_sources(_coll, removed)

            changed = [doc for doc in docs if state.get(doc.source) != new_state[doc.source]]

            if not changed:
                all_docs = _coll.get()
                if all_docs and all_docs.get("documents"):
                    _bm25_corpus = list(all_docs["documents"])
                    from rank_bm25 import BM25Okapi
                    import jieba
                    _bm25_index = BM25Okapi([list(jieba.cut(d)) for d in _bm25_corpus])
                _ready = True
                logger.info("Memory index is up to date (%d chunks)", _coll.count())
                return True

            delete_sources(_coll, [doc.source for doc in changed])
            total = 0
            for doc in changed:
                chunks = split_text(doc.text)
                ids, texts, metas = [], [], []
                for c in chunks:
                    ids.append(c.chunk_id)
                    texts.append(c.text)
                    metas.append({"source": doc.source, "heading": c.heading or ""})
                if ids:
                    _coll.add(ids=ids, documents=texts, metadatas=metas)
                    total += len(ids)

            db = _coll.get()
            if db and db.get("documents"):
                _bm25_corpus = list(db["documents"])
                import jieba
                from rank_bm25 import BM25Okapi
                _bm25_index = BM25Okapi([list(jieba.cut(d)) for d in _bm25_corpus])

            save_state(new_state)
            logger.info("Vectorized %d chunks from %d sources", total, len(changed))
            _ready = True
            return True
        except Exception as e:
            logger.warning("Memory init deferred: %s", e)
            return False


def recall(query: str, budget_chars: int = 2000, top_k: int = 8) -> str:
    """Search local ChromaDB + BM25. Returns joined text or '' on failure."""
    q = (query or "").strip()
    if len(q) < MIN_QUERY_CHARS:
        return ""
    if not _ensure():
        return ""
    try:
        results = _coll.query(query_texts=[q], n_results=top_k)
        seen, chunks = set(), []

        if results.get("documents"):
            for doc in results["documents"][0] if results["documents"] else []:
                if doc and doc not in seen:
                    seen.add(doc)
                    chunks.append(doc)

        if _bm25_index:
            import jieba
            scores = _bm25_index.get_scores(list(jieba.cut(q)))
            best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            for idx in best:
                if idx < len(_bm25_corpus) and _bm25_corpus[idx] not in seen:
                    seen.add(_bm25_corpus[idx])
                    chunks.append(_bm25_corpus[idx])

        joined = "\n\n".join(chunks)
        if len(joined) > budget_chars:
            joined = joined[:budget_chars] + "..."
        return joined
    except Exception as e:
        logger.warning("Memory search failed: %s", e)
        return ""
