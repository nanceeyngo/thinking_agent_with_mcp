"""
analysis_dashboard/retrieval_tools.py

LangChain @tool implementations for the Log Analysis Agent.

Tools

  search_logs_semantic    - cosine-similarity vector search over the
                            LangGraph SQLite log store.
  list_recent_logs        - time-ordered scan of log entries.
  get_session_trace       - retrieve all entries for a specific session_id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from typing import Any

from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.sqlite.aio import AsyncSqliteStore

from analysis_dashboard.settings import settings

logger = logging.getLogger("analysis_agent")


# Persistent background event loop


class _BackgroundLoop:
    """
    A single daemon thread that owns one asyncio event loop for the entire
    process lifetime.  All coroutines are submitted via run() and block the
    calling thread until the result (or exception) is available.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="retrieval-async-loop",
            daemon=True,  # dies with the main process; no cleanup needed
        )
        self._thread.start()

    def _run_forever(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro):
        """Submit *coro* to the background loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()  # re-raises exceptions from the coroutine


# Module-level singleton — created once when the module is first imported.
_bg = _BackgroundLoop()


def run_in_background(coro):
    """Public helper: run *coro* in the persistent background loop."""
    return _bg.run(coro)


# Store singleton — lives inside the background loop's thread

_store: AsyncSqliteStore | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_store_cm = None  # context-manager reference so __aexit__ can be called


async def _get_store_async() -> AsyncSqliteStore:
    """
    Return the shared AsyncSqliteStore, creating it on first call.
    Must be awaited *inside* the background loop.
    """
    global _store, _embeddings, _store_cm

    if _store is not None:
        return _store

    #  embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    #  ensure DB file exists
    db_path = settings.log_db_path
    if not os.path.exists(db_path):
        logger.warning("Database file not found: %s — creating empty file.", db_path)
        conn = sqlite3.connect(db_path)
        conn.close()

    #  open the store inside the background loop (correct loop binding)
    _store_cm = AsyncSqliteStore.from_conn_string(
        db_path,
        index={
            "embed": _embeddings,
            "dims": 384,  # all-MiniLM-L6-v2
            "fields": ["content"],
        },
    )
    _store = await _store_cm.__aenter__()
    logger.info("AsyncSqliteStore initialised: %s", db_path)
    return _store


async def _search_async(
    namespace_prefix: tuple[str, ...],
    query: str = "",
    limit: int = 30,
):
    store = await _get_store_async()
    return await store.asearch(namespace_prefix, query=query, limit=limit)


# Public sync helpers (called from Streamlit / sync tool wrappers)


def search_sync(
    namespace: tuple[str, ...],
    query: str,
    limit: int = 30,
):
    """Blocking search — safe to call from any thread / sync context."""
    return run_in_background(_search_async(namespace, query, limit))


# Shared helper


# def _item_to_dict(item) -> dict[str, Any]:
#     value = dict(item.value) if hasattr(item.value, "items") else {}
#     return {
#         "key": item.key,
#         "namespace": list(item.namespace) if hasattr(item, "namespace") else [],
#         "score": getattr(item, "score", None),
#         "session_id": value.get("session_id", ""),
#         "mcp_interaction_type": value.get("mcp_interaction_type", ""),
#         "component": value.get("component", ""),
#         "content": value.get("content", ""),
#         "timestamp": value.get("timestamp", 0.0),
#         "metadata": value.get("metadata", {}),
#     }


def _sanitize_for_json(text: str) -> str:
    """
    Remove control characters and escape special chars that break JSON strings.
    """
    if not text:
        return ""
    # Remove null bytes and other control characters except \n, \r, \t
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")
    # Truncate to avoid massive payloads
    return text


# def _item_to_dict(item, content_limit: int = 200) -> dict[str, Any]:
#     """
#     Convert a store item to a compact dict safe to pass between LLM tool calls.
#     content_limit controls how many characters of 'content' are kept — keeping
#     this small prevents the LLM from truncating JSON when forwarding payloads.
#     """
#     value = dict(item.value) if hasattr(item.value, "items") else {}
#     # Keep only essential metadata keys to reduce token count
#     raw_meta = value.get("metadata", {})
#     slim_meta = {
#         k: raw_meta[k]
#         for k in (
#             "tool_name",
#             "latency_ms",
#             "token_count",
#             "response_length",
#             "error",
#             "level",
#         )
#         if k in raw_meta
#     }
#     return {
#         "key": item.key,
#         "session_id": value.get("session_id", ""),
#         "mcp_interaction_type": value.get("mcp_interaction_type", ""),
#         "component": value.get("component", ""),
#         "content": value.get("content", "")[:content_limit],
#         "timestamp": value.get("timestamp", 0.0),
#         "metadata": slim_meta,
#     }


def _item_to_dict(
    item, content_limit: int = 500, full_output: bool = False
) -> dict[str, Any]:
    """
    Convert a store item to a compact dict safe to pass between LLM tool calls.

    Args:
        item: Store item from LangGraph
        content_limit: Max characters for content field (ignored if full_output=True)
        full_output: If True, include full content (for session trace pass-through)
    """
    value = dict(item.value) if hasattr(item.value, "items") else {}

    # Keep only essential metadata keys to reduce token count
    raw_meta = value.get("metadata", {})
    slim_meta = {
        k: raw_meta[k]
        for k in (
            "tool_name",
            "latency_ms",
            "token_count",
            "response_length",
            "token_usage_source",
            "error",
            "level",
            "request_id",
            "prompt_tokens",
            "completion_tokens",
            "message_count",  # For request entries
            "max_tokens",
        )
        if k in raw_meta
    }

    # Sanitize content to prevent JSON breaking
    raw_content = value.get("content", "")
    sanitized_content = _sanitize_for_json(raw_content)

    limit = (
        len(sanitized_content)
        if full_output
        else min(len(sanitized_content), content_limit)
    )

    return {
        "key": item.key,
        "session_id": value.get("session_id", ""),
        "mcp_interaction_type": value.get("mcp_interaction_type", ""),
        "component": value.get("component", ""),
        "content": sanitized_content[:limit],
        "timestamp": value.get("timestamp", 0.0),
        "metadata": slim_meta,
    }


# LangChain tools


@tool
def search_logs_semantic(
    query: str,
    namespace_prefix: str = "logs",
    limit: int = 15,
) -> str:
    """
    Perform semantic vector-similarity search over stored MCP execution logs.

    This tool embeds the query using the same embedding model used during
    ingestion and retrieves the most semantically similar log entries from
    the LangGraph SQLite vector store.

    Use this to:
      - Find sessions where a specific tool failed.
      - Discover latency anomalies or error patterns.
      - Locate sampling requests related to a specific topic.
      - Identify unmapped system errors across execution histories.

    Args:
        query:            Natural-language search query.
        namespace_prefix: Restrict search scope. Use dot-separated path,
                          e.g. "logs.mcp.client" or just "logs" for all.
        limit:            Maximum number of results (default 15).

    Returns:
        JSON string containing a list of matching log entries with
        relevance scores, session IDs, and full content.
    """
    logger.info(
        "search_logs_semantic | query='%s' | prefix='%s'", query, namespace_prefix
    )
    try:
        ns_tuple = tuple(namespace_prefix.split(".")) if namespace_prefix else ("logs",)
        results = run_in_background(_search_async(ns_tuple, query, limit))
        items = [_item_to_dict(r) for r in results]
        items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        logger.info("search_logs_semantic returned %d results", len(items))
        return json.dumps(items, indent=2, default=str, ensure_ascii=False)
    except Exception as exc:
        logger.error("search_logs_semantic error: %s", exc)
        return json.dumps({"error": str(exc), "results": []})


@tool
def list_recent_logs(
    namespace_prefix: str = "logs",
    limit: int = 30,
) -> str:
    """
    List the most recently stored log entries from the execution store.

    Unlike search_logs_semantic (which ranks by semantic similarity),
    this tool retrieves entries in insertion order — useful for seeing
    what happened in chronological sequence.

    Args:
        namespace_prefix: Dot-separated namespace to scan, e.g. "logs.mcp".
        limit:            Maximum entries to return (default 30).

    Returns:
        JSON string with a time-ordered list of log entries.
    """
    logger.info("list_recent_logs | prefix='%s' | limit=%d", namespace_prefix, limit)
    try:
        ns_tuple = tuple(namespace_prefix.split(".")) if namespace_prefix else ("logs",)
        results = run_in_background(_search_async(ns_tuple, "", limit))
        items = [_item_to_dict(r) for r in results]
        items.sort(key=lambda x: x.get("timestamp", 0.0), reverse=True)
        logger.info("list_recent_logs returned %d entries", len(items))
        return json.dumps(items, indent=2, default=str, ensure_ascii=False)
    except Exception as exc:
        logger.error("list_recent_logs error: %s", exc)
        return json.dumps({"error": str(exc), "results": []})


@tool
def get_session_trace(session_id: str) -> str:
    """
    Retrieve all log entries for a specific session_id, sorted by timestamp.

    A session_id uniquely identifies a single run of the MCP client agent.
    This tool returns the full causal trace — from initial query to final
    corrected answer — for that session.

    Args:
        session_id: UUID string identifying the target session.

    Returns:
        JSON string with all log entries for that session, sorted by
        timestamp ascending.
    """
    logger.info("get_session_trace | session_id=%s", session_id)
    try:
        results = run_in_background(
            _search_async(("logs",), f"session_id {session_id}", 100)
        )
        # items = [
        #     _item_to_dict(r) for r in results
        # if r.value.get("session_id") == session_id
        # ]
        items = [
            _item_to_dict(
                r, full_output=True
            )  # Include full content for session traces
            for r in results
            if r.value.get("session_id") == session_id
        ]
        items.sort(key=lambda x: x.get("timestamp", 0.0))
        logger.info(
            "get_session_trace found %d entries for session %s", len(items), session_id
        )
        return json.dumps(items, indent=2, default=str, ensure_ascii=False)
    except Exception as exc:
        logger.error("get_session_trace error: %s", exc)
        return json.dumps({"error": str(exc), "entries": []})


# Graceful shutdown (optional — call from app teardown if needed)


async def _close_store_async():
    global _store, _store_cm
    if _store is not None and _store_cm is not None:
        try:
            await _store_cm.__aexit__(None, None, None)
            logger.info("AsyncSqliteStore closed.")
        except Exception as exc:
            logger.error("Error closing AsyncSqliteStore: %s", exc)
        finally:
            _store = None
            _store_cm = None


def close_store():
    """Sync wrapper — call from atexit or Streamlit teardown."""
    run_in_background(_close_store_async())


# """
# analysis_dashboard/retrieval_tools.py

# LangChain @tool implementations for the Log Analysis Agent.

# Tools

#   search_logs_semantic    - cosine-similarity vector search over the
#                             LangGraph SQLite log store.
#   list_recent_logs        - time-ordered scan of log entries.
#   get_session_trace       - retrieve all entries for a specific session_id.
# """

# from __future__ import annotations

# import asyncio
# import json
# import logging
# import os
# import sqlite3
# from typing import Any

# from langchain.tools import tool
# from langchain_huggingface import HuggingFaceEmbeddings
# from langgraph.store.sqlite.aio import AsyncSqliteStore

# from analysis_dashboard.settings import settings

# logger = logging.getLogger("analysis_agent")


# # Store access helpers - using AsyncSqliteStore
# _store = None
# _embeddings = None
# _store_cm = None  # Context manager reference for cleanup
# _store_lock = asyncio.Lock()  # Thread safety


# # --- Helper: Safe Async Execution ---
# def _run_sync(coro):
#     """
#     Runs a coroutine safely. If an event loop is already running,
#     we use run_coroutine_threadsafe or similar. If not, we use asyncio.run().
#     """
#     try:
#         # Check if there is an existing event loop
#         loop = asyncio.get_running_loop()
#         # If a loop is running, we cannot start a new one,
#         # so we schedule the coroutine in the existing loop.
#         if loop.is_running():
#             future = asyncio.run_coroutine_threadsafe(coro, loop)
#             return future.result()
#     except RuntimeError:
#         # No loop is running, safe to start a new one
#         return asyncio.run(coro)


# async def _get_store_async():
#     """Get or create the AsyncSqliteStore singleton."""
#     global _store, _embeddings, _store_cm

#     if _store is None:
#         # Initialize embeddings once
#         if _embeddings is None:
#             _embeddings = HuggingFaceEmbeddings(
#                 model_name=settings.embedding_model_name,
#                 model_kwargs={"device": "cpu"},
#                 encode_kwargs={"normalize_embeddings": True},
#             )

#         # Ensure the database file exists
#         db_path = settings.log_db_path
#         if not os.path.exists(db_path):
#             logger.warning(
#                 f"Database file not found: {db_path}. Creating new database."
#             )
#             # Note: This is an exception to async usage, but it is a
#               file creation check
#             # that happens once.
#             conn = sqlite3.connect(db_path)
#             conn.close()
#             logger.info(f"Created new database file: {db_path}")

#         try:
#             # Use AsyncSqliteStore.from_conn_string (same as client)
#             _store_cm = AsyncSqliteStore.from_conn_string(
#                 db_path,  # Direct path like the client
#                 index={
#                     "embed": _embeddings,
#                     "dims": 384,  # all-MiniLM-L6-v2 dimension
#                     "fields": ["content"],
#                 },
#             )
#             _store = await _store_cm.__aenter__()
#             logger.info("AsyncSqliteStore initialized with path: %s", db_path)
#         except Exception as e:
#             logger.error("Failed to initialize AsyncSqliteStore: %s", e)
#             raise

#     return _store


# # def _get_store_sync():
# #     """Synchronous wrapper for getting the store (for use in sync tools)."""
# #     try:
# #         loop = asyncio.get_event_loop()
# #         if loop.is_running():
# #             # We're in an async context, but LangChain tools are sync
# #             # Create a new event loop for this operation
# #             new_loop = asyncio.new_event_loop()
# #             try:
# #                 return new_loop.run_until_complete(_get_store_async())
# #             finally:
# #                 new_loop.close()
# #         else:
# #             return loop.run_until_complete(_get_store_async())
# #     except RuntimeError:
# #         return asyncio.run(_get_store_async())


# async def _search_async(
#     namespace_prefix: tuple[str, ...], query: str = "", limit: int = 30
# ):
#     """Async search wrapper."""
#     try:
#         store = await _get_store_async()
#         results = await store.asearch(
#             namespace_prefix,
#             query=query,
#             limit=limit,
#         )
#         return results
#     except Exception as e:
#         logger.error("Error occurred while searching: %s", e)
#         raise


# def search_sync(namespace: tuple, query: str, limit: int = 30):
#     """Synchronous version of search for UI use."""
#     return _run_sync(_search_async(namespace, query, limit))


# # def _run_async_in_new_loop(coro):
# #     """Run an async coroutine in a new event loop (thread-safe)."""
# #     loop = asyncio.new_event_loop()
# #     try:
# #         return loop.run_until_complete(coro)
# #     finally:
# #         # Clean up pending tasks
# #         pending = asyncio.all_tasks(loop)
# #         for task in pending:
# #             task.cancel()
# #         loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
# #         loop.close()


# # def _search_sync(namespace_prefix, query="", limit=30):
# #     """Synchronous search wrapper."""
# #     try:
# #         return _run_async_in_new_loop(_search_async(namespace_prefix, query, limit))

# #     except RuntimeError:
# #         return asyncio.run(_search_async(namespace_prefix, query, limit))


# # Utility: flatten store item to dict
# def _item_to_dict(item) -> dict[str, Any]:
#     """Convert a store item to a dictionary."""
#     value = dict(item.value) if hasattr(item.value, "items") else {}
#     return {
#         "key": item.key,
#         "namespace": list(item.namespace) if hasattr(item, "namespace") else [],
#         "score": getattr(item, "score", None),
#         "session_id": value.get("session_id", ""),
#         "mcp_interaction_type": value.get("mcp_interaction_type", ""),
#         "component": value.get("component", ""),
#         "content": value.get("content", ""),
#         "timestamp": value.get("timestamp", 0.0),
#         "metadata": value.get("metadata", {}),
#     }


# # @tools
# @tool
# def search_logs_semantic(
#     query: str,
#     namespace_prefix: str = "logs",
#     limit: int = 15,
# ) -> str:
#     """
#     Perform semantic vector-similarity search over stored MCP execution logs.

#     This tool embeds the query using the same embedding model used during
#     ingestion and retrieves the most semantically similar log entries from
#     the LangGraph SQLite vector store.

#     Use this to:
#       - Find sessions where a specific tool failed.
#       - Discover latency anomalies or error patterns.
#       - Locate sampling requests related to a specific topic.
#       - Identify unmapped system errors across execution histories.

#     Args:
#         query:            Natural-language search query.
#         namespace_prefix: Restrict search scope. Use dot-separated path,
#                           e.g. "logs.mcp.client" or just "logs" for all.
#         limit:            Maximum number of results (default 15).

#     Returns:
#         JSON string containing a list of matching log entries with
#         relevance scores, session IDs, and full content.
#     """
#     logger.info(
#         "search_logs_semantic | query='%s' | prefix='%s'", query, namespace_prefix
#     )
#     try:
#         # Convert dot-separated prefix string to tuple
#         ns_tuple = tuple(namespace_prefix.split(".")) if namespace_prefix
# else ("logs",)

#         async def task():
#             results = await _search_async(ns_tuple, query, limit)
#             # results = _search_sync(ns_tuple, query, limit)
#             items = [_item_to_dict(r) for r in results]

#             # Sort by relevance score if available
#             items.sort(key=lambda x: x.get("score", 0.0), reverse=True)

#             logger.info("search_logs_semantic returned %d results", len(items))
#             return json.dumps(items, indent=2, default=str)

#         return _run_sync(task())

#     except Exception as exc:
#         logger.error("search_logs_semantic error: %s", exc)
#         return json.dumps({"error": str(exc), "results": []})


# @tool
# async def list_recent_logs(
#     namespace_prefix: str = "logs",
#     limit: int = 30,
# ) -> str:
#     """
#     List the most recently stored log entries from the execution store.

#     Unlike search_logs_semantic (which ranks by semantic similarity),
#     this tool retrieves entries in insertion order — useful for seeing
#     what happened in chronological sequence.

#     Args:
#         namespace_prefix: Dot-separated namespace to scan, e.g. "logs.mcp".
#         limit:            Maximum entries to return (default 30).

#     Returns:
#         JSON string with a time-ordered list of log entries.
#     """
#     logger.info("list_recent_logs | prefix='%s' | limit=%d", namespace_prefix, limit)
#     try:
#         ns_tuple = tuple(namespace_prefix.split(".")) if namespace_prefix
# else ("logs",)

#         # Empty-string query returns items ordered by recency
#         # results = _search_sync(ns_tuple, "", limit)
#         # Await search directly
#         async def task():
#             results = await _search_async(ns_tuple, "", limit)
#             items = [_item_to_dict(r) for r in results]

#             # Sort by timestamp descending (most recent first)
#             items.sort(key=lambda x: x.get("timestamp", 0.0), reverse=True)

#             logger.info("list_recent_logs returned %d entries", len(items))
#             return json.dumps(items, indent=2, default=str)

#         return _run_sync(task())

#     except Exception as exc:
#         logger.error("list_recent_logs error: %s", exc)
#         return json.dumps({"error": str(exc), "results": []})


# @tool
# async def get_session_trace(session_id: str) -> str:
#     """
#     Retrieve all log entries for a specific session_id, sorted by timestamp.

#     A session_id uniquely identifies a single run of the MCP client agent.
#     This tool returns the full causal trace — from initial query to final
#     corrected answer — for that session.

#     Args:
#         session_id: UUID string identifying the target session.

#     Returns:
#         JSON string with all log entries for that session, sorted by
#         timestamp ascending.
#     """
#     logger.info("get_session_trace | session_id=%s", session_id)
#     try:

#         # Search with the session_id as the query to find all related entries
#         # The session_id is stored in both the content and metadata fields
#         # results = _search_sync(("logs",), f"session_id {session_id}", 100)
#         # Await search directly
#         async def task():
#             results = await _search_async(("logs",), f"session_id {session_id}", 100)
#             # Filter strictly to the requested session
#             items = [
#                 _item_to_dict(r)
#                 for r in results
#                 if r.value.get("session_id") == session_id
#             ]
#             return items

#         items = _run_sync(task())

#         # Sort by timestamp ascending to show causal order
#         items.sort(key=lambda x: x.get("timestamp", 0.0))

#         logger.info(
#             "get_session_trace found %d entries for session %s",
# len(items), session_id
#         )
#         return json.dumps(items, indent=2, default=str)

#     except Exception as exc:
#         logger.error("get_session_trace error: %s", exc)
#         return json.dumps({"error": str(exc), "entries": []})


# # Cleanup function for graceful shutdown
# async def close_store():
#     """Gracefully close the AsyncSqliteStore connection."""
#     global _store, _store_cm
#     if _store is not None and _store_cm is not None:
#         try:
#             await _store_cm.__aexit__(None, None, None)
#             logger.info("AsyncSqliteStore connection closed")
#         except Exception as e:
#             logger.error("Error closing AsyncSqliteStore: %s", e)
#         finally:
#             _store = None
#             _store_cm = None
