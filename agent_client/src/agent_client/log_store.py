"""
agent_client/log_store.py

LangGraph SQLite vector log store.

Key responsibilities:
  • Instantiate langgraph.store.sqlite.SqliteStore backed by a real
    OpenAI text-embedding-3-small embedding model so every log entry
    is stored with a semantic vector and is searchable.
  • Expose write_log() — persists a validated LogEntry into a
    hierarchical dot-separated namespace tuple.
  • Expose search_logs() — performs cosine-similarity vector search
    over all stored entries and returns the top-k matches.

Namespace hierarchy examples

  ("logs", "agent", "planning", "reflexive_loop")
  ("logs", "mcp", "server", "tools", "execute_code")
  ("logs", "mcp", "client", "tool_call", "reflect_and_correct")
  ("logs", "mcp", "client", "resource_read", "domain_docs")
  ("logs", "mcp", "sampling", "request")

Data-guardrail: Any integer keys produced by JSON round-tripping are
cast back to int before leaving this module (JSON converts int keys
to strings; we reverse that transformation here).
"""

from __future__ import annotations

import time
import uuid
from typing import Any
import atexit
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.base import Item
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Settings


class StoreSettings(BaseSettings):
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    log_db_path: str = "mcp_agent_log.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


_store_settings = StoreSettings()  # type: ignore[call-arg]


# 2. Validated log-entry schema


class LogEntry(BaseModel):
    """
    Validated schema for a single log trace entry.

    Fields
    ------
    session_id          Unique UUID tracking the multi-turn execution trace.
    mcp_interaction_type
        Explicitly typed interaction kind:
          • "tool_invocation"    — client called a remote MCP tool
          • "resource_read"      — client read a remote MCP resource
          • "sampling_request"   — server delegated an LLM call to client
          • "agent_action"       — internal agent reasoning step
          • "tool_observation"   — result returned from a tool call
    component           Dot-path of the originating component,
                        e.g. "agent.planning.reflexive_loop"
    content             Raw text payload targeted for vector index search.
    metadata            Optional extra key-value pairs (latency_ms,
                        token_count, tool_name, …).  Integer keys
                        survive round-trips intact.
    timestamp           Unix epoch float — set automatically.
    """

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    mcp_interaction_type: str  # one of the typed strings above
    component: str = "agent"  # dot-separated component path
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    def to_store_value(self) -> dict[str, Any]:
        """
        Serialize to the dict stored in the LangGraph store value.
        Re-cast any str keys that should be int (JSON round-trip guard).
        """
        raw = self.model_dump()
        raw["metadata"] = _recast_int_keys(raw.get("metadata", {}))
        return raw

    @classmethod
    def from_store_item(cls, item: Item) -> "LogEntry":
        """Deserialise from a LangGraph store Item."""
        value = item.value
        if "metadata" in value:
            value["metadata"] = _recast_int_keys(value["metadata"])
        return cls(**value)


def _recast_int_keys(d: dict) -> dict:
    """
    Recursively cast string keys that look like integers back to int.

    JSON serialisation converts integer dict-keys to strings:
        {0: "a"} → {"0": "a"}
    This guard reverses that transformation so callers receive the
    original types.
    """
    result: dict = {}
    for k, v in d.items():
        try:
            new_key: Any = (
                int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k
            )
        except (ValueError, TypeError):
            new_key = k
        result[new_key] = _recast_int_keys(v) if isinstance(v, dict) else v
    return result


# 3. Namespace helpers


def component_to_namespace(component: str) -> tuple[str, ...]:
    """
    Convert a dot-separated component path to a LangGraph namespace tuple.

    "agent.planning.reflexive_loop"
        → ("logs", "agent", "planning", "reflexive_loop")

    "mcp.server.tools.execute_code"
        → ("logs", "mcp", "server", "tools", "execute_code")
    """
    parts = component.strip(".").split(".")
    return ("logs",) + tuple(parts)


# 4. Singleton store management - THIS IS THE KEY FIX
class StoreManager:
    """Manages a single persistent store connection."""

    def __init__(self):
        self._store = None
        self._store_cm = None
        self._embedding_model = None
        self._closed = False

    async def get_store(self):
        """Get or create the store connection."""
        if self._closed:
            raise RuntimeError("Store has been closed")

        if self._store is None:
            # Create embedding model once
            if self._embedding_model is None:
                self._embedding_model = HuggingFaceEmbeddings(
                    model_name=_store_settings.embedding_model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )

            # Import and create store
            from langgraph.store.sqlite.aio import AsyncSqliteStore

            import os

            db_path = _store_settings.log_db_path
            if not os.path.exists(db_path):
                import sqlite3

                conn = sqlite3.connect(db_path)
                conn.close()

            self._store_cm = AsyncSqliteStore.from_conn_string(
                db_path,
                index={
                    "embed": self._embedding_model,
                    "dims": 384,  # FIXED: all-MiniLM-L6-v2 outputs 384 dimensions
                    "fields": ["content"],
                },
            )
            self._store = await self._store_cm.__aenter__()

        return self._store

    async def close(self):
        """Properly close the store connection."""
        if self._store is not None and not self._closed:
            try:
                await self._store_cm.__aexit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._store = None
                self._closed = True


# Global store manager instance
_store_manager = StoreManager()


# Register cleanup on exit
def _cleanup():
    """Synchronous cleanup for atexit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_store_manager.close())
        else:
            loop.run_until_complete(_store_manager.close())
    except Exception:
        pass


atexit.register(_cleanup)

# 5. Public API - FIXED


async def write_log(entry: LogEntry) -> str:
    """
    Persist a LogEntry to the vector store.

    Returns the key under which the entry was stored.
    """
    store = await _store_manager.get_store()
    namespace = component_to_namespace(entry.component)
    key = str(uuid.uuid4())
    value = entry.to_store_value()

    try:
        await store.aput(namespace=namespace, key=key, value=value)
        return key
    except Exception as e:
        # Log but don't crash
        print(f"[LOG_STORE ERROR] Failed to write log: {e}")
        return ""


async def search_logs(
    query: str,
    *,
    namespace_prefix: tuple[str, ...] = ("logs",),
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Semantic vector search over stored log entries."""
    try:
        store = await _store_manager.get_store()
        results = await store.asearch(
            namespace_prefix,
            query=query,
            limit=limit,
        )
        output: list[dict[str, Any]] = []
        for item in results:
            value = item.value
            if "metadata" in value:
                value["metadata"] = _recast_int_keys(value["metadata"])
            output.append(
                {
                    "key": item.key,
                    "namespace": item.namespace,
                    "score": getattr(item, "score", None),
                    "value": value,
                }
            )
        return output
    except Exception as e:
        print(f"[LOG_STORE ERROR] Search failed: {e}")
        return []


async def list_recent_logs(
    namespace_prefix: tuple[str, ...] = ("logs",),
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List recent log entries without semantic scoring."""
    try:
        store = await _store_manager.get_store()
        results = await store.asearch(
            namespace_prefix,
            query="",
            limit=limit,
        )
        output: list[dict[str, Any]] = []
        for item in results:
            value = item.value
            if "metadata" in value:
                value["metadata"] = _recast_int_keys(value["metadata"])
            output.append(
                {
                    "key": item.key,
                    "namespace": item.namespace,
                    "score": getattr(item, "score", None),
                    "value": value,
                }
            )
        return output
    except Exception as e:
        print(f"[LOG_STORE ERROR] List failed: {e}")
        return []


# Optional: Add cleanup function for graceful shutdown
async def close_store():
    """Gracefully close the store connection."""
    await _store_manager.close()
