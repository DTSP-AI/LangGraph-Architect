# memory_manager.py

import os
import json
import time
import logging
import pickle
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

from dotenv import load_dotenv

# ─── LangChain & Vector Imports ─────────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.chat_history import InMemoryChatMessageHistory

# ─── Environment & Logger Setup ─────────────────────────────────────────────────
load_dotenv()

# Vector DB config (may be absent in env)
PGVECTOR_CONN_STR    = os.getenv("PGVECTOR_CONNECTION_STRING")
COLLECTION_NAME      = os.getenv("PGVECTOR_COLLECTION_NAME", "ai_pipeline_memory")
DECAY_RATE           = float(os.getenv("VECTOR_DECAY_RATE", "0.01"))
RETRIEVAL_K          = int(os.getenv("VECTOR_RETRIEVAL_K", "6"))
VECTOR_TTL_DAYS      = int(os.getenv("VECTOR_TTL_DAYS", "30"))

# Chat history persistence
HISTORY_DIR = os.getenv("CHAT_HISTORY_DIR", os.path.join(os.getcwd(), "data", "history"))
os.makedirs(HISTORY_DIR, exist_ok=True)

# Feedback log path
LOG_DIR  = os.getenv("FEEDBACK_LOG_DIR", os.path.join(os.getcwd(), "data"))
LOG_PATH = os.path.join(LOG_DIR, "feedback_log.json")
os.makedirs(LOG_DIR, exist_ok=True)

# Caps
_MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "100"))

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─── In-Memory Chat History w/ Disk Persistence ─────────────────────────────────
_session_histories: Dict[str, InMemoryChatMessageHistory] = {}

def _history_filepath(session_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{session_id}.pkl")

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Load or create the session's InMemoryChatMessageHistory.
    Auto-prunes old messages and persists to disk.
    """
    if not session_id or not isinstance(session_id, str):
        raise ValueError("session_id must be a non-empty string")

    history = _session_histories.get(session_id)
    if history is None:
        # Attempt to load from disk
        path = _history_filepath(session_id)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    history = pickle.load(f)
                logger.info(f"Loaded chat history from disk for '{session_id}'")
            except Exception as e:
                logger.error(f"Failed to load chat history for '{session_id}': {e}")
        if history is None:
            history = InMemoryChatMessageHistory()
            logger.info(f"Created new chat history for '{session_id}'")
        _session_histories[session_id] = history

    # Prune if too long
    if hasattr(history, "messages") and len(history.messages) > _MAX_HISTORY_LENGTH:
        history.messages = history.messages[-_MAX_HISTORY_LENGTH:]
        logger.info(f"Pruned chat history for '{session_id}' to {_MAX_HISTORY_LENGTH} messages")

    return history

def clear_chat_history(session_id: str) -> None:
    """
    Clear both in-memory and on-disk history for a session.
    """
    _session_histories.pop(session_id, None)
    path = _history_filepath(session_id)
    if os.path.exists(path):
        try:
            os.remove(path)
            logger.info(f"Deleted chat history file for '{session_id}'")
        except Exception as e:
            logger.error(f"Failed to delete chat history file for '{session_id}': {e}")

def _persist_chat_history(session_id: str) -> None:
    """
    Internal: write the session history to disk.
    """
    history = _session_histories.get(session_id)
    if history:
        try:
            with open(_history_filepath(session_id), "wb") as f:
                pickle.dump(history, f)
            logger.debug(f"Persisted chat history for '{session_id}'")
        except Exception as e:
            logger.error(f"Failed to persist chat history for '{session_id}': {e}")

# ─── Persistent Vector Memory (PGVector) + TTL ─────────────────────────────────
_vector_retriever: Optional[TimeWeightedVectorStoreRetriever] = None
_embedding_model = OpenAIEmbeddings()

def init_vector_retriever() -> Optional[TimeWeightedVectorStoreRetriever]:
    """
    Initialize PGVector + TimeWeightedVectorStoreRetriever with retry & seeding.
    Returns None if PGVECTOR_CONN_STR is not configured.
    """
    global _vector_retriever
    if not PGVECTOR_CONN_STR:
        logger.warning("PGVECTOR_CONNECTION_STRING not set; skipping vector memory setup")
        return None

    if _vector_retriever is None:
        last_exc = None
        for attempt in range(5):
            try:
                store = PGVector(
                    connection_string=PGVECTOR_CONN_STR,
                    collection_name=COLLECTION_NAME,
                    embedding_function=_embedding_model
                )
                # Seed if empty
                hits = store.similarity_search("init", k=1)
                if not hits:
                    seed = Document(
                        page_content="Seed document for PGVector init",
                        metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
                    )
                    store.add_documents([seed])
                    logger.info(f"Seeded vector collection '{COLLECTION_NAME}'")
                _vector_retriever = TimeWeightedVectorStoreRetriever(
                    vectorstore=store,
                    decay_rate=DECAY_RATE,
                    k=RETRIEVAL_K,
                    other_score_keys=["importance"]
                )
                logger.info("Initialized TimeWeightedVectorStoreRetriever")
                break
            except Exception as e:
                last_exc = e
                logger.warning(f"PGVector init attempt {attempt+1}/5 failed: {e}")
                time.sleep(2 ** attempt)
        else:
            logger.critical("Unable to initialize PGVector retriever after retries")
            raise RuntimeError("PGVector initialization failed") from last_exc
    return _vector_retriever

def get_vector_retriever() -> Optional[TimeWeightedVectorStoreRetriever]:
    return init_vector_retriever()

def add_to_vector_memory(content: str, metadata: Optional[dict] = None) -> None:
    """
    Add a new Document to the vector store, stamping it with a timestamp.
    No-ops if vector retriever is unavailable.
    """
    retriever = get_vector_retriever()
    if retriever is None:
        return

    try:
        meta = (metadata.copy() if metadata else {})
        meta["timestamp"] = datetime.now(timezone.utc).isoformat()
        doc = Document(page_content=content, metadata=meta)
        retriever.vectorstore.add_documents([doc])
        logger.info("Added document to vector memory")
    except Exception as e:
        logger.error(f"Error adding to vector memory: {e}", exc_info=True)

def query_vector_memory(query: str, k: Optional[int] = None) -> List[Document]:
    """
    Retrieve relevant docs, filtering out those older than VECTOR_TTL_DAYS.
    Returns an empty list if vector retriever is unavailable.
    """
    retriever = get_vector_retriever()
    if retriever is None:
        return []

    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        logger.error(f"Error querying vector memory: {e}", exc_info=True)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=VECTOR_TTL_DAYS)
    fresh = []
    for doc in docs:
        ts = doc.metadata.get("timestamp")
        try:
            dt = datetime.fromisoformat(ts)
            if dt >= cutoff:
                fresh.append(doc)
        except Exception:
            # No or invalid timestamp → include by default
            fresh.append(doc)
    result = fresh[:k] if (k is not None and k < len(fresh)) else fresh
    logger.debug(f"Returning {len(result)} documents from vector memory (TTL filtered)")
    return result

# ─── Feedback Logging to JSON File ───────────────────────────────────────────────
def log_feedback(workflow_id: str, approved: bool, comments: str = "") -> dict:
    """
    Append a feedback entry with timestamp into feedback_log.json.
    """
    entry = {
        "workflow_id": workflow_id,
        "approved":    approved,
        "comments":    comments,
        "timestamp":   datetime.now(timezone.utc).isoformat()
    }
    data = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read feedback log: {e}")
    data.append(entry)
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write feedback log: {e}")
    logger.info(f"Logged feedback: {entry}")
    return {"status": "logged", "entry": entry}

def get_feedback() -> List[dict]:
    """
    Return all feedback entries, or an empty list if none.
    """
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read feedback log: {e}")
        return []
