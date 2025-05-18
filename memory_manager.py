# memory_manager.py

import os
import json
import logging
import pickle
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

from dotenv import load_dotenv
from sqlalchemy.orm import declarative_base

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.chat_history import InMemoryChatMessageHistory

# ─── Load Env ─────────────────────────────────────────────────────
load_dotenv()

PGVECTOR_CONN_STR    = os.getenv("PGVECTOR_CONNECTION_STRING")
COLLECTION_NAME      = os.getenv("PGVECTOR_COLLECTION_NAME", "ai_pipeline_memory")
DECAY_RATE           = float(os.getenv("VECTOR_DECAY_RATE", "0.01"))
RETRIEVAL_K          = int(os.getenv("VECTOR_RETRIEVAL_K", "6"))
VECTOR_TTL_DAYS      = int(os.getenv("VECTOR_TTL_DAYS", "30"))

HISTORY_DIR          = os.getenv("CHAT_HISTORY_DIR", os.path.join(os.getcwd(), "data", "history"))
LOG_DIR              = os.getenv("FEEDBACK_LOG_DIR", os.path.join(os.getcwd(), "data"))
LOG_PATH             = os.path.join(LOG_DIR, "feedback_log.json")
_MAX_HISTORY_LENGTH  = int(os.getenv("MAX_HISTORY_LENGTH", "100"))

os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

# ─── Alembic Anchor ───────────────────────────────────────────────
Base = declarative_base()

class VectorMemory(Base):
    __tablename__ = "vector_memory"
    id = ...

# ─── In-Memory Chat History ───────────────────────────────────────
_session_histories: Dict[str, InMemoryChatMessageHistory] = {}

def _history_filepath(session_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{session_id}.pkl")

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if not session_id:
        raise ValueError("Invalid session_id")

    history = _session_histories.get(session_id)
    if history is None:
        path = _history_filepath(session_id)
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    history = pickle.load(f)
                if not isinstance(history, InMemoryChatMessageHistory):
                    history = InMemoryChatMessageHistory()
            else:
                history = InMemoryChatMessageHistory()
        except Exception as e:
            logger.warning(f"Failed to load chat history for {session_id}: {e}")
            history = InMemoryChatMessageHistory()
        _session_histories[session_id] = history

    if len(history.messages) > _MAX_HISTORY_LENGTH:
        history.messages = history.messages[-_MAX_HISTORY_LENGTH:]
        _persist_chat_history(session_id)

    return history

def _persist_chat_history(session_id: str) -> None:
    history = _session_histories.get(session_id)
    if history:
        try:
            with open(_history_filepath(session_id), "wb") as f:
                pickle.dump(history, f)
        except Exception as e:
            logger.error(f"Failed to persist chat history: {e}")

def clear_chat_history(session_id: str) -> None:
    _session_histories.pop(session_id, None)
    try:
        os.remove(_history_filepath(session_id))
    except FileNotFoundError:
        pass

def add_message_to_history(session_id: str, content: str, role: str = "human") -> None:
    history = get_chat_history(session_id)
    if role == "ai":
        history.add_ai_message(content)
    else:
        history.add_user_message(content)
    _persist_chat_history(session_id)

# ─── PGVector with TimeWeighted Retriever ─────────────────────────
_vector_retriever: Optional[TimeWeightedVectorStoreRetriever] = None
_embedding_model: Optional[OpenAIEmbeddings] = None

def _get_embedding_model() -> OpenAIEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OpenAIEmbeddings()
    return _embedding_model

def init_vector_retriever() -> Optional[TimeWeightedVectorStoreRetriever]:
    global _vector_retriever
    if not PGVECTOR_CONN_STR:
        logger.warning("PGVECTOR_CONNECTION_STRING is missing")
        return None

    if _vector_retriever is None:
        try:
            embeddings = _get_embedding_model()
            store = PGVector(
                connection_string=PGVECTOR_CONN_STR,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings
            )
            _vector_retriever = TimeWeightedVectorStoreRetriever(
                vectorstore=store,
                decay_rate=DECAY_RATE,
                k=RETRIEVAL_K,
                other_score_keys=["importance"]
            )
            logger.info(f"Initialized PGVector for {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"PGVector init failed: {e}", exc_info=True)

    return _vector_retriever

def get_vector_retriever() -> Optional[TimeWeightedVectorStoreRetriever]:
    if _vector_retriever is None:
        return init_vector_retriever()
    return _vector_retriever

def add_to_vector_memory(content: str, metadata: Optional[Dict] = None) -> None:
    retriever = get_vector_retriever()
    if retriever is None:
        return
    metadata = metadata or {}
    metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    metadata.setdefault("importance", 1.0)
    doc = Document(page_content=content, metadata=metadata)
    retriever.vectorstore.add_documents([doc])

def query_vector_memory(query: str, k: Optional[int] = None) -> List[Document]:
    retriever = get_vector_retriever()
    if retriever is None:
        return []

    docs = retriever.get_relevant_documents(query)
    cutoff = datetime.now(timezone.utc) - timedelta(days=VECTOR_TTL_DAYS)

    filtered = []
    for doc in docs:
        ts = doc.metadata.get("timestamp")
        try:
            if ts and datetime.fromisoformat(ts) >= cutoff:
                filtered.append(doc)
        except Exception:
            continue
    return filtered[: (k or RETRIEVAL_K)]

# ─── Feedback Log ────────────────────────────────────────────────
def log_feedback(workflow_id: str, approved: bool, comments: str = "") -> dict:
    entry = {
        "workflow_id": workflow_id,
        "approved": approved,
        "comments": comments,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    data = []
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.append(entry)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Feedback logging failed: {e}")
    return {"status": "logged", "entry": entry}

def get_feedback() -> List[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
