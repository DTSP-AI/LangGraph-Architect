# memory_manager.py

import os
import json
import time
import logging
import pickle
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # Added BaseMessage, HumanMessage, AIMessage

# ─── Environment & Logger Setup ─────────────────────────────────────────────────
# ... (rest of the setup is the same) ...
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
# logger.setLevel(logging.INFO) # This should be set by the application importing it or via global logging config

# ─── In-Memory Chat History w/ Disk Persistence ─────────────────────────────────
_session_histories: Dict[str, InMemoryChatMessageHistory] = {}

def _history_filepath(session_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{session_id}.pkl")

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if not session_id or not isinstance(session_id, str):
        logger.error("get_chat_history called with invalid session_id.") # Added logging
        raise ValueError("session_id must be a non-empty string")

    history = _session_histories.get(session_id)
    if history is None:
        path = _history_filepath(session_id)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    history = pickle.load(f)
                if not isinstance(history, InMemoryChatMessageHistory): # Basic check
                    logger.warning(f"Loaded object for session '{session_id}' is not InMemoryChatMessageHistory. Creating new.")
                    history = InMemoryChatMessageHistory()
                else:
                    logger.info(f"Loaded chat history from disk for '{session_id}'")
            except Exception as e:
                logger.error(f"Failed to load chat history for '{session_id}': {e}. Creating new.")
                history = InMemoryChatMessageHistory() # Ensure history is always an object
        else:
            history = InMemoryChatMessageHistory()
            logger.info(f"Created new chat history for '{session_id}'")
        _session_histories[session_id] = history

    if hasattr(history, "messages") and len(history.messages) > _MAX_HISTORY_LENGTH:
        history.messages = history.messages[-_MAX_HISTORY_LENGTH:]
        logger.info(f"Pruned chat history for '{session_id}' to {_MAX_HISTORY_LENGTH} messages")
        _persist_chat_history(session_id) # Persist after pruning

    return history

def clear_chat_history(session_id: str) -> None:
    if not session_id or not isinstance(session_id, str): # Added validation
        logger.warning("clear_chat_history called with invalid session_id.")
        return

    _session_histories.pop(session_id, None)
    path = _history_filepath(session_id)
    if os.path.exists(path):
        try:
            os.remove(path)
            logger.info(f"Deleted chat history file for '{session_id}'")
        except Exception as e:
            logger.error(f"Failed to delete chat history file for '{session_id}': {e}")

def _persist_chat_history(session_id: str) -> None:
    history = _session_histories.get(session_id)
    if history: # Only persist if history exists in memory cache
        try:
            with open(_history_filepath(session_id), "wb") as f:
                pickle.dump(history, f)
            logger.debug(f"Persisted chat history for '{session_id}'")
        except Exception as e:
            logger.error(f"Failed to persist chat history for '{session_id}': {e}")

def add_lc_message_to_history(session_id: str, message: BaseMessage) -> None:
    """Adds a Langchain BaseMessage to the history and persists."""
    if not isinstance(message, BaseMessage):
        logger.error(f"Attempted to add non-BaseMessage to history for session '{session_id}'. Type: {type(message)}")
        return # Or raise error
    history = get_chat_history(session_id)
    history.add_message(message)
    _persist_chat_history(session_id)

def add_messages_to_history(session_id: str, messages: List[BaseMessage]) -> None:
    """Adds a list of Langchain BaseMessages to the history and persists."""
    history = get_chat_history(session_id)
    valid_messages_added = 0
    for msg in messages:
        if isinstance(msg, BaseMessage):
            history.add_message(msg)
            valid_messages_added += 1
        else:
            logger.warning(f"Skipped non-BaseMessage during bulk add to history for session '{session_id}'. Type: {type(msg)}")
    if valid_messages_added > 0:
        _persist_chat_history(session_id)

# Deprecating the old string-based add_message_to_history or making it adapt
def add_message_to_history(session_id: str, message_content: str, role: str = "human") -> None:
    """
    Adds a message to history. If role is 'human', adds HumanMessage. If 'ai', adds AIMessage.
    For simple string messages. Consider using add_lc_message_to_history for BaseMessage objects.
    """
    history = get_chat_history(session_id)
    if role == "human":
        history.add_user_message(message_content) # Equivalent to HumanMessage
    elif role == "ai":
        history.add_ai_message(message_content) # Equivalent to AIMessage
    else:
        # Fallback or specific handling for generic "message"
        # For simplicity, let's assume it's a generic string to be added if not human/ai
        # However, InMemoryChatMessageHistory expects HumanMessage or AIMessage primarily.
        # It's better to be explicit.
        logger.warning(f"Generic message role '{role}' for session '{session_id}'. Consider using 'human' or 'ai'.")
        history.add_message(HumanMessage(content=f"[{role}] {message_content}")) # Example: prefixing role

    _persist_chat_history(session_id)


# ─── Persistent Vector Memory (PGVector) + TTL ─────────────────────────────────
_vector_retriever: Optional[TimeWeightedVectorStoreRetriever] = None
_embedding_model: Optional[OpenAIEmbeddings] = None # Initialize later

def _get_embedding_model() -> OpenAIEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = OpenAIEmbeddings()
    return _embedding_model

def init_vector_retriever() -> Optional[TimeWeightedVectorStoreRetriever]:
    global _vector_retriever
    if not PGVECTOR_CONN_STR:
        logger.warning("PGVECTOR_CONNECTION_STRING not set; skipping vector memory setup")
        return None

    if _vector_retriever is None: # Attempt to initialize only if not already done
        logger.info("Attempting to initialize TimeWeightedVectorStoreRetriever...")
        last_exc = None
        for attempt in range(3): # Reduced retries for quicker feedback
            try:
                current_embeddings = _get_embedding_model()
                store = PGVector(
                    connection_string=PGVECTOR_CONN_STR,
                    collection_name=COLLECTION_NAME,
                    embedding_function=current_embeddings
                )
                # Seed if empty - this can be slow, consider if truly needed on every init
                # For now, let's assume seeding is a one-time or rare operation,
                # or handled by a separate script. For web app, better to be fast.
                # If you must seed:
                # try:
                #     hits = store.similarity_search("init_check_doc", k=1) # Check with a specific query
                #     if not hits:
                #         seed_doc = Document(
                #             page_content="Initial document to ensure collection exists and is queryable.",
                #             metadata={"timestamp": datetime.now(timezone.utc).isoformat(), "type": "system_seed"}
                #         )
                #         store.add_documents([seed_doc])
                #         logger.info(f"Seeded vector collection '{COLLECTION_NAME}' as it appeared empty.")
                # except Exception as seed_exc:
                # logger.warning(f"Could not check or seed vector store (continuing): {seed_exc}")

                _vector_retriever = TimeWeightedVectorStoreRetriever(
                    vectorstore=store,
                    decay_rate=DECAY_RATE,
                    k=RETRIEVAL_K,
                    other_score_keys=["importance"] # Ensure documents have an 'importance' metadata if used
                )
                logger.info(f"Initialized TimeWeightedVectorStoreRetriever for collection '{COLLECTION_NAME}'.")
                break # Success
            except Exception as e:
                last_exc = e
                logger.warning(f"PGVector init attempt {attempt+1}/3 failed: {e}")
                time.sleep(1 + attempt) # Shorter sleep for web app
        else: # If loop finished without break
            logger.error(f"Unable to initialize PGVector retriever after retries: {last_exc}", exc_info=True)
            # Not raising error here, so app can continue without vector memory if needed.
            # UI should reflect if vector memory is unavailable.
            # raise RuntimeError("PGVector initialization failed") from last_exc
    return _vector_retriever

def get_vector_retriever() -> Optional[TimeWeightedVectorStoreRetriever]:
    """Returns the retriever if initialized, otherwise tries to initialize it."""
    if _vector_retriever is None:
        return init_vector_retriever() # Attempt to initialize on first call
    return _vector_retriever

def add_to_vector_memory(content: str, metadata: Optional[dict] = None) -> None:
    retriever = get_vector_retriever()
    if retriever is None:
        logger.warning("Vector retriever not available. Cannot add to vector memory.")
        return

    try:
        meta = (metadata.copy() if metadata else {})
        meta.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        # Add default importance if not present, as TimeWeightedVectorStoreRetriever might use it
        meta.setdefault("importance", 1.0) # Default importance

        doc = Document(page_content=content, metadata=meta)
        retriever.vectorstore.add_documents([doc])
        logger.info(f"Added document to vector memory. Session ID in metadata: {meta.get('session_id')}")
    except Exception as e:
        logger.error(f"Error adding to vector memory: {e}", exc_info=True)

def query_vector_memory(query: str, k: Optional[int] = None, filter_metadata: Optional[Dict] = None) -> List[Document]:
    retriever = get_vector_retriever()
    if retriever is None:
        logger.warning("Vector retriever not available. Cannot query vector memory.")
        return []

    try:
        # Note: TimeWeightedVectorStoreRetriever's get_relevant_documents doesn't directly accept a 'filter' argument.
        # Filtering by metadata often needs to happen at the underlying vectorstore level if supported,
        # or by post-processing the results.
        # For PGVector, the `similarity_search` method of the store itself can take a `filter` dict.

        # If session-specific filtering is essential for TimeWeighted:
        # 1. Fetch more docs (e.g., k * num_sessions_expected)
        # 2. Manually filter by session_id from metadata
        # 3. Then apply time weighting logic (which TimeWeightedVectorStoreRetriever does internally)
        # This is complex. Simpler for now: TimeWeighted is global, but documents *can* have session_id.

        # For now, let's assume query is global and then filtered by TTL
        # If you need to pass a filter to PGVector directly (bypassing some TimeWeighted logic):
        # docs = retriever.vectorstore.similarity_search(query, k=(k or RETRIEVAL_K), filter=filter_metadata)

        docs = retriever.get_relevant_documents(query) # This applies time weighting
        logger.debug(f"Retrieved {len(docs)} docs before TTL filter for query: '{query}'")

    except Exception as e:
        logger.error(f"Error querying vector memory: {e}", exc_info=True)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=VECTOR_TTL_DAYS)
    fresh_docs = []
    for doc in docs:
        ts_str = doc.metadata.get("timestamp")
        try:
            if ts_str:
                dt = datetime.fromisoformat(ts_str)
                # Ensure tz-aware comparison if dt is naive (though isoformat() should include tz)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc) # Assume UTC if naive
                if dt >= cutoff:
                    # If session-specific filtering is also required here (post-retrieval)
                    if filter_metadata and filter_metadata.get("session_id"):
                        if doc.metadata.get("session_id") == filter_metadata["session_id"]:
                            fresh_docs.append(doc)
                    else: # No session filter, just TTL
                        fresh_docs.append(doc)
            else: # No timestamp, include by default (or decide to exclude)
                if filter_metadata and filter_metadata.get("session_id"):
                    if doc.metadata.get("session_id") == filter_metadata["session_id"]:
                        fresh_docs.append(doc) # No TTL check, but session matches
                else:
                    fresh_docs.append(doc) # Include if no timestamp and no session filter

        except ValueError: # Invalid timestamp format
            logger.warning(f"Document with invalid timestamp format: {ts_str}. Including by default.")
            if filter_metadata and filter_metadata.get("session_id"):
                 if doc.metadata.get("session_id") == filter_metadata["session_id"]:
                    fresh_docs.append(doc)
            else:
                fresh_docs.append(doc)


    # Apply k limit after all filtering
    final_k = k if k is not None else RETRIEVAL_K
    result = fresh_docs[:final_k]

    logger.debug(f"Returning {len(result)} documents from vector memory (TTL and session filtered if applicable)")
    return result

# ─── Feedback Logging to JSON File ───────────────────────────────────────────────
# ... (log_feedback and get_feedback are fine as they are) ...
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