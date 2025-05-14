# memory_manager.py

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict

from dotenv import load_dotenv

# ─── LangChain Imports ─────────────────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.chat_history import InMemoryChatMessageHistory

##############################################
# Environment & Logger Setup
##############################################
load_dotenv()

# Vector DB environment
PGVECTOR_CONN_STR    = os.getenv("PGVECTOR_CONNECTION_STRING")
COLLECTION_NAME      = os.getenv("PGVECTOR_COLLECTION_NAME", "ai_pipeline_memory")
DECAY_RATE           = float(os.getenv("VECTOR_DECAY_RATE", "0.01"))
RETRIEVAL_K          = int(os.getenv("VECTOR_RETRIEVAL_K", "6"))

if not PGVECTOR_CONN_STR:
    raise RuntimeError("PGVECTOR_CONNECTION_STRING environment variable is not set")

# Feedback log path
LOG_DIR   = os.getenv("FEEDBACK_LOG_DIR", os.path.join(os.getcwd(), "data"))
LOG_PATH  = os.path.join(LOG_DIR, "feedback_log.json")

# Chat history caps
_MAX_HISTORY_LENGTH = 100

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

##############################################
# In-Memory Chat History Management
##############################################
_session_histories: Dict[str, InMemoryChatMessageHistory] = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retrieve or create the session's chat history.
    Auto-prunes messages exceeding _MAX_HISTORY_LENGTH.
    """
    if not session_id or not isinstance(session_id, str):
        logger.error("Invalid session_id for chat history retrieval.")
        raise ValueError("session_id must be a non-empty string")

    history = _session_histories.get(session_id)
    if history is None:
        logger.info(f"Creating new chat history for session '{session_id}'")
        history = InMemoryChatMessageHistory()
        _session_histories[session_id] = history

    # Prune if too long
    if len(history.messages) > _MAX_HISTORY_LENGTH:
        logger.info(f"Pruning chat history for session '{session_id}' to last {_MAX_HISTORY_LENGTH} messages")
        history.messages = history.messages[-_MAX_HISTORY_LENGTH:]

    return history

def clear_chat_history(session_id: str) -> None:
    """
    Clear the chat history for a session.
    """
    if session_id in _session_histories:
        logger.info(f"Clearing chat history for session '{session_id}'")
        del _session_histories[session_id]
    else:
        logger.debug(f"No chat history to clear for session '{session_id}'")

##############################################
# Persistent Vector Memory (pgvector) Setup
##############################################
_vector_retriever: Optional[TimeWeightedVectorStoreRetriever] = None
_embedding_model = OpenAIEmbeddings()

def init_vector_retriever() -> TimeWeightedVectorStoreRetriever:
    """
    Connect to a persistent PGVector store and return a TimeWeightedVectorStoreRetriever.
    Retries on failure and seeds the collection if empty.
    """
    global _vector_retriever
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
                    logger.info(f"Seeding PGVector collection '{COLLECTION_NAME}'")
                    seed_doc = Document(page_content="Seed entry for initialization")
                    store.add_documents([seed_doc])
                _vector_retriever = TimeWeightedVectorStoreRetriever(
                    vectorstore=store,
                    decay_rate=DECAY_RATE,
                    k=RETRIEVAL_K,
                    other_score_keys=["importance"]
                )
                logger.info("Vector retriever initialized")
                break
            except Exception as e:
                last_exc = e
                logger.warning(f"PGVector init attempt {attempt+1}/5 failed: {e}")
                time.sleep(2 ** attempt)
        else:
            logger.critical("Failed to initialize PGVector retriever after retries")
            raise RuntimeError("Could not connect to PGVector") from last_exc
    return _vector_retriever

def get_vector_retriever() -> TimeWeightedVectorStoreRetriever:
    """
    Return the cached vector retriever, initializing if needed.
    """
    return init_vector_retriever()

def add_to_vector_memory(content: str, metadata: Optional[dict] = None) -> None:
    """
    Embed and store a document in the vector database.
    """
    retriever = get_vector_retriever()
    try:
        doc = Document(page_content=content, metadata=metadata or {})
        retriever.vectorstore.add_documents([doc])
        logger.info("Added document to vector memory")
    except Exception as e:
        logger.error(f"Error adding to vector memory: {e}", exc_info=True)

def query_vector_memory(query: str, k: Optional[int] = None) -> List[Document]:
    """
    Retrieve relevant documents from the vector database.
    """
    retriever = get_vector_retriever()
    docs = retriever.get_relevant_documents(query)
    return docs[:k] if (k is not None and k < len(docs)) else docs

##############################################
# Feedback Logging
##############################################
def _ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

def log_feedback(workflow_id: str, approved: bool, comments: str = "") -> dict:
    """
    Stores feedback with timestamp into a local JSON file.
    """
    _ensure_log_dir()
    entry = {
        "workflow_id": workflow_id,
        "approved": approved,
        "comments": comments,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    data = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read existing feedback log: {e}")

    data.append(entry)
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write feedback log: {e}")

    logger.info(f"Feedback logged: {entry}")
    return {"status": "logged", "entry": entry}

def get_feedback() -> List[dict]:
    """
    Retrieves all feedback entries, or empty list if none exist.
    """
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read feedback log: {e}")
        return []
