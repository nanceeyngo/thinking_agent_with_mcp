"""
analysis_dashboard/neo4j_tools.py

LangChain @tools for Neo4j Aura DB knowledge graph operations.

Graph Schema

Nodes
  (:Session   {session_id, started_at, query_count})
  (:AgentAction {action_id, session_id, component, content,
                 timestamp, mcp_interaction_type})
  (:MCPServerCall {call_id, session_id, tool_name, latency_ms,
                   timestamp, mcp_interaction_type})

Edges
  (:Session)-[:TRIGGERED]->(:AgentAction)
  (:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)
  (:MCPServerCall)-[:DEPENDS_ON]->(:AgentAction)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from langchain.tools import tool
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from analysis_dashboard.settings import settings

logger = logging.getLogger("analysis_agent")


# Shared background loop - imported from retrieval_tools so there is only
# ONE loop in the whole process.


def _run_sync(coro):
    """Submit *coro* to the persistent background loop and block until done."""
    from analysis_dashboard.retrieval_tools import run_in_background

    return run_in_background(coro)


# Neo4j driver singleton

_driver = None
_neo4j_available = None
_db_name = None


def _get_database_name(driver):
    """Auto-detect the available database name."""
    try:
        with driver.session() as session:
            result = session.run("SHOW DATABASES")
            databases = [record["name"] for record in result]
            logger.info("Available databases: %s", databases)
            if not databases:
                logger.error("No databases found")
                return None
            return "neo4j" if "neo4j" in databases else databases[0]
    except Exception as exc:
        logger.warning("Could not list databases: %s", exc)
        return None


def _get_driver():
    """Get or create Neo4j driver with auto-detection of database name."""
    global _driver, _neo4j_available, _db_name

    if _neo4j_available is False:
        return None

    if _driver is not None:
        return _driver

    if not settings.neo4j_uri or not settings.neo4j_password:
        logger.warning("Neo4j credentials not configured")
        _neo4j_available = False
        return None

    try:
        password = (
            settings.neo4j_password.get_secret_value()
            if settings.neo4j_password
            else ""
        )

        base_uri = (
            settings.neo4j_uri.replace("neo4j+s://", "")
            .replace("neo4j+ssc://", "")
            .replace("bolt://", "")
            .replace("neo4j://", "")
        )
        if ":" in base_uri:
            base_uri = base_uri.split(":")[0]

        uris_to_try = list(
            dict.fromkeys(
                [
                    settings.neo4j_uri,
                    f"neo4j+ssc://{base_uri}",
                    f"bolt://{base_uri}:7687",
                ]
            )
        )

        for uri in uris_to_try:
            driver = None
            try:
                logger.info("Trying Neo4j connection: %s…", uri[:50])
                driver = GraphDatabase.driver(
                    uri,
                    auth=(settings.neo4j_username, password),
                    connection_timeout=15,
                    max_connection_lifetime=3600,
                )
                _db_name = _get_database_name(driver)
                if not _db_name:
                    logger.warning(
                        "Could not detect database name, defaulting to 'neo4j'"
                    )
                    _db_name = "neo4j"

                with driver.session(database=_db_name) as session:
                    session.run("RETURN 1 as test", timeout=10).single()

                logger.info("Neo4j connected — database: '%s' via %s", _db_name, uri)
                _driver = driver
                _neo4j_available = True
                return _driver

            except (ServiceUnavailable, Exception) as exc:
                logger.warning("Failed with %s: %s", uri, exc)
                if driver:
                    driver.close()

        logger.error("All Neo4j connection attempts failed")
        _neo4j_available = False

    except Exception as exc:
        logger.error("Neo4j driver creation failed: %s", exc)
        _neo4j_available = False

    return None


# Cypher execution helpers


async def _run_cypher_async(
    query: str, params: dict | None = None
) -> list[dict[str, Any]]:
    """
    Execute a Cypher query on a thread so the blocking driver call never
    stalls the background event loop.
    """

    def _execute():
        driver = _get_driver()
        if driver is None:
            return []
        try:
            with driver.session(database=_db_name) as session:
                result = session.run(query, params or {}, timeout=30)
                return [dict(record) for record in result]
        except Exception as exc:
            logger.error("Cypher query failed: %s", exc)
            return []

    return await asyncio.to_thread(_execute)


def run_cypher_sync(query: str, params: dict | None = None) -> list[dict[str, Any]]:
    """Synchronous wrapper for UI / non-async callers."""
    return _run_sync(_run_cypher_async(query, params))


# Schema bootstrap


def ensure_schema() -> None:
    """Create uniqueness constraints if they don't exist yet."""

    async def _internal():
        driver = _get_driver()
        if driver is None:
            logger.warning("Neo4j not available — skipping schema creation")
            return

        constraints = [
            (
                "CREATE CONSTRAINT session_id_unique IF NOT EXISTS "
                "FOR (s:Session) REQUIRE s.session_id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT agent_action_id_unique IF NOT EXISTS "
                "FOR (a:AgentAction) REQUIRE a.action_id IS UNIQUE"
            ),
            (
                "CREATE CONSTRAINT mcp_call_id_unique IF NOT EXISTS "
                "FOR (m:MCPServerCall) REQUIRE m.call_id IS UNIQUE"
            ),
        ]
        for constraint in constraints:
            try:
                await _run_cypher_async(constraint)
                logger.debug("Constraint applied.")
            except Exception as exc:
                logger.debug("Constraint may already exist: %s", exc)

        logger.info("Neo4j schema ensured on database '%s'", _db_name)

    _run_sync(_internal())


# LangChain tools


@tool
def project_session_to_graph(session_json: str) -> str:
    """
    Project a full session execution trace into the Neo4j Aura DB
    knowledge graph.

    Extracts client-side AgentAction nodes and server-side MCPServerCall
    nodes from the trace, then links them with typed directional edges:
      (:Session)-[:TRIGGERED]->(:AgentAction)
      (:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)
      (:MCPServerCall)-[:DEPENDS_ON]->(:AgentAction)

    Args:
        session_json: JSON string — the output of get_session_trace tool,
                      containing a list of log entry dicts.

    Returns:
        JSON summary of nodes and edges written to Neo4j.
    """
    logger.info("project_session_to_graph called")

    async def _task():
        entries: list[dict[str, Any]] = json.loads(session_json)
        if not entries:
            return json.dumps({"status": "no_entries", "nodes": 0, "edges": 0})

        session_id = entries[0].get("session_id", str(uuid.uuid4()))
        nodes_created = 0
        edges_created = 0

        # 1. Upsert Session node
        query_count = sum(
            1
            for e in entries
            if e.get("component", "").startswith("agent.planning.query")
        )
        started_at = min((e.get("timestamp", 0.0) for e in entries), default=0.0)

        await _run_cypher_async(
            """
            MERGE (s:Session {session_id: $session_id})
            ON CREATE SET s.started_at = $started_at, s.query_count = $query_count
            ON MATCH  SET s.query_count = $query_count
            """,
            {
                "session_id": session_id,
                "started_at": started_at,
                "query_count": query_count,
            },
        )
        nodes_created += 1

        # 2. Process each log entry
        prev_agent_action_id: str | None = None

        for entry in entries:
            interaction_type = entry.get("mcp_interaction_type", "")
            component = entry.get("component", "")
            content = entry.get("content", "")[:500]
            timestamp = entry.get("timestamp", 0.0)
            metadata = entry.get("metadata", {})

            if interaction_type in ("agent_action", "tool_observation"):
                action_id = f"{session_id}:{entry.get('key', str(uuid.uuid4()))}"
                await _run_cypher_async(
                    """
                    MERGE (a:AgentAction {action_id: $action_id})
                    ON CREATE SET
                        a.session_id           = $session_id,
                        a.component            = $component,
                        a.content              = $content,
                        a.timestamp            = $timestamp,
                        a.mcp_interaction_type = $interaction_type
                    """,
                    {
                        "action_id": action_id,
                        "session_id": session_id,
                        "component": component,
                        "content": content,
                        "timestamp": timestamp,
                        "interaction_type": interaction_type,
                    },
                )
                nodes_created += 1

                await _run_cypher_async(
                    """
                    MATCH (s:Session {session_id: $session_id})
                    MATCH (a:AgentAction {action_id: $action_id})
                    MERGE (s)-[:TRIGGERED]->(a)
                    """,
                    {"session_id": session_id, "action_id": action_id},
                )
                edges_created += 1
                prev_agent_action_id = action_id

            elif interaction_type in (
                "tool_invocation",
                "resource_read",
                "sampling_request",
            ):
                call_id = f"{session_id}:{entry.get('key', str(uuid.uuid4()))}"
                tool_name = metadata.get("tool_name", component.split(".")[-1])
                latency_ms = int(metadata.get("latency_ms", 0))

                await _run_cypher_async(
                    """
                    MERGE (m:MCPServerCall {call_id: $call_id})
                    ON CREATE SET
                        m.session_id           = $session_id,
                        m.tool_name            = $tool_name,
                        m.latency_ms           = $latency_ms,
                        m.timestamp            = $timestamp,
                        m.content              = $content,
                        m.mcp_interaction_type = $interaction_type
                    """,
                    {
                        "call_id": call_id,
                        "session_id": session_id,
                        "tool_name": tool_name,
                        "latency_ms": latency_ms,
                        "timestamp": timestamp,
                        "content": content,
                        "interaction_type": interaction_type,
                    },
                )
                nodes_created += 1

                if prev_agent_action_id:
                    await _run_cypher_async(
                        """
                        MATCH (a:AgentAction {action_id: $action_id})
                        MATCH (m:MCPServerCall {call_id: $call_id})
                        MERGE (a)-[:ROUTED_TO]->(m)
                        """,
                        {"action_id": prev_agent_action_id, "call_id": call_id},
                    )
                    edges_created += 1

                await _run_cypher_async(
                    """
                    MATCH (m:MCPServerCall {call_id: $call_id})
                    MATCH (s:Session {session_id: $session_id})
                    MERGE (m)-[:DEPENDS_ON]->(s)
                    """,
                    {"call_id": call_id, "session_id": session_id},
                )
                edges_created += 1

        result = {
            "status": "success",
            "session_id": session_id,
            "nodes_created": nodes_created,
            "edges_created": edges_created,
        }
        logger.info("project_session_to_graph: %s", result)
        return json.dumps(result, indent=2)

    try:
        return _run_sync(_task())
    except Exception as exc:
        logger.error("project_session_to_graph error: %s", exc)
        return json.dumps({"status": "error", "message": str(exc)})


@tool
def query_knowledge_graph(cypher_query: str) -> str:
    """
    Execute a read-only Cypher query against the Neo4j Aura DB graph
    and return the results as JSON.

    Useful for:
      - Inspecting which sessions are stored in the graph.
      - Tracing the path from a Session through AgentActions to MCPServerCalls.
      - Finding MCPServerCalls with high latency.
      - Checking graph topology.

    Example queries:
      MATCH (s:Session) RETURN s.session_id, s.query_count LIMIT 10
      MATCH (s:Session)-[:TRIGGERED]->(a:AgentAction) RETURN s,a LIMIT 5
      MATCH (m:MCPServerCall) WHERE m.latency_ms > 2000 RETURN m

    Args:
        cypher_query: A valid Cypher read query string.

    Returns:
        JSON string with query results (list of row dicts).
    """
    logger.info("query_knowledge_graph | query='%s'", cypher_query[:120])

    async def _task():
        rows = await _run_cypher_async(cypher_query)
        clean_rows = []
        for row in rows:
            clean_row = {}
            for k, v in row.items():
                try:
                    json.dumps(v)
                    clean_row[k] = v
                except TypeError:
                    clean_row[k] = str(v)
            clean_rows.append(clean_row)
        logger.info("query_knowledge_graph returned %d rows", len(clean_rows))
        return json.dumps(clean_rows, indent=2, default=str)

    try:
        return _run_sync(_task())
    except Exception as exc:
        logger.error("query_knowledge_graph error: %s", exc)
        return json.dumps({"error": str(exc)})


@tool
def get_graph_summary() -> str:
    """
    Return a high-level summary of the current state of the Neo4j
    knowledge graph: node counts, edge counts, and the most recent sessions.

    No arguments required.

    Returns:
        JSON with node/edge counts and recent session list.
    """
    logger.info("get_graph_summary called")

    async def _task():
        counts, edges, recent_sessions = await asyncio.gather(
            _run_cypher_async(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count "
                "ORDER BY count DESC"
            ),
            _run_cypher_async(
                "MATCH ()-[r]->() RETURN type(r) AS relationship, count(r) AS count "
                "ORDER BY count DESC"
            ),
            _run_cypher_async(
                "MATCH (s:Session) "
                "RETURN s.session_id AS session_id, "
                "       s.query_count AS query_count, "
                "       s.started_at  AS started_at "
                "ORDER BY s.started_at DESC LIMIT 5"
            ),
        )
        return json.dumps(
            {
                "node_counts": counts,
                "edge_counts": edges,
                "recent_sessions": recent_sessions,
            },
            indent=2,
            default=str,
        )

    try:
        return _run_sync(_task())
    except Exception as exc:
        logger.error("get_graph_summary error: %s", exc)
        return json.dumps({"error": str(exc)})


# """
# analysis_dashboard/neo4j_tools.py

# LangChain @tools for Neo4j Aura DB knowledge graph operations.

# Graph Schema

# Nodes
#   (:Session   {session_id, started_at, query_count})
#   (:AgentAction {action_id, session_id, component, content,
#                  timestamp, mcp_interaction_type})
#   (:MCPServerCall {call_id, session_id, tool_name, latency_ms,
#                    timestamp, mcp_interaction_type})

# Edges
#   (:Session)-[:TRIGGERED]->(:AgentAction)
#   (:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)
#   (:MCPServerCall)-[:DEPENDS_ON]->(:AgentAction)

# """

# from __future__ import annotations

# import json
# import logging
# import asyncio
# import uuid
# from typing import Any

# from langchain.tools import tool
# from neo4j import GraphDatabase
# from neo4j.exceptions import ServiceUnavailable

# from analysis_dashboard.settings import settings

# logger = logging.getLogger("analysis_agent")

# # Neo4j driver singleton

# # _driver = None


# # def _get_driver():
# #     global _driver
# #     if _driver is None:
# #         from neo4j import GraphDatabase
# #         from neo4j.exceptions import ServiceUnavailable

# #         try:
# #             if not settings.neo4j_uri or not settings.neo4j_password:
# #                 logger.warning(
# #                     "Neo4j credentials not configured. Graph features disabled."
# #                 )
# #                 return None

# #             password = (
# #                 settings.neo4j_password.get_secret_value()
# #                 if settings.neo4j_password
# #                 else ""
# #             )

# #             logger.info(f"Connecting to Neo4j at {settings.neo4j_uri[:30]}...")

# #             _driver = GraphDatabase.driver(
# #                 settings.neo4j_uri,
# #                 auth=(settings.neo4j_username, password),
# #                 # Add connection timeout
# #                 connection_timeout=10,
# #                 # Increase max connection lifetime
# #                 max_connection_lifetime=3600,
# #             )

# #             # Test connection with timeout
# #             with _driver.session(database="neo4j") as session:
# #                 result = session.run("RETURN 1 as test", timeout=5)
# #                 result.single()

# #             logger.info("Neo4j connection established successfully")

# #         except ServiceUnavailable as e:
# #             logger.error(f"Neo4j service unavailable: {e}")
# #             logger.error(f"Check that the URI '{settings.neo4j_uri}' is correct")
# #             _driver = None
# #         except Exception as e:
# #             logger.error(f"Failed to connect to Neo4j: {e}")
# #             _driver = None

# #     return _driver


# # def _get_driver():
# #     global _driver
# #     if _driver is None:
# #         from neo4j import GraphDatabase  # type: ignore[import]

# #         password = (
# #             settings.neo4j_password.get_secret_value()
# #             if settings.neo4j_password
# #             else ""
# #         )
# #         _driver = GraphDatabase.driver(
# #             settings.neo4j_uri,
# #             auth=(settings.neo4j_username, password),
# #         )
# #     return _driver


# # def _run_cypher(query: str, params: dict | None = None) -> list[dict[str, Any]]:
# #     """Execute a Cypher query and return rows as list of dicts."""
# #     driver = _get_driver()
# #     with driver.session() as session:
# #         result = session.run(query, params or {})
# #         return [dict(record) for record in result]


# # --- Add this helper at the top (or near your other helpers) ---
# def _run_sync(coro):
#     """Bridge for running async logic in synchronous tool definitions."""
#     try:
#         loop = asyncio.get_running_loop()
#         if loop.is_running():
#             return asyncio.run_coroutine_threadsafe(coro, loop).result()
#     except RuntimeError:
#         return asyncio.run(coro)


# # Neo4j driver singleton
# _driver = None
# _neo4j_available = None
# _db_name = None  # Will store the actual database name


# def _get_database_name(driver):
#     """Auto-detect the available database name."""
#     try:
#         with driver.session() as session:
#             # List all databases
#             result = session.run("SHOW DATABASES")
#             databases = [record["name"] for record in result]
#             logger.info(f"Available databases: {databases}")

#             if not databases:
#                 logger.error("No databases found")
#                 return None

#             # Prefer 'neo4j' if it exists, otherwise use the first one
#             if "neo4j" in databases:
#                 return "neo4j"
#             return databases[0]
#     except Exception as e:
#         logger.warning(f"Could not list databases: {e}")
#         return None


# def _get_driver():
#     """Get or create Neo4j driver with auto-detection of database name."""
#     global _driver, _neo4j_available, _db_name

#     if _neo4j_available is False:
#         return None

#     if _driver is None:
#         if not settings.neo4j_uri or not settings.neo4j_password:
#             logger.warning("Neo4j credentials not configured")
#             _neo4j_available = False
#             return None

#         try:
#             password = (
#                 settings.neo4j_password.get_secret_value()
#                 if settings.neo4j_password
#                 else ""
#             )

#             # Try different URI schemes
#             base_uri = (
#                 settings.neo4j_uri.replace("neo4j+s://", "")
#                 .replace("neo4j+ssc://", "")
#                 .replace("bolt://", "")
#                 .replace("neo4j://", "")
#             )
#             if ":" in base_uri:
#                 base_uri = base_uri.split(":")[0]

#             uris_to_try = [
#                 settings.neo4j_uri,
#                 f"neo4j+ssc://{base_uri}",
#                 f"bolt://{base_uri}:7687",
#             ]
#             uris_to_try = list(dict.fromkeys(uris_to_try))  # Remove duplicates

#             for uri in uris_to_try:
#                 try:
#                     logger.info(f"Trying connection: {uri[:50]}...")

#                     driver = GraphDatabase.driver(
#                         uri,
#                         auth=(settings.neo4j_username, password),
#                         connection_timeout=15,
#                         max_connection_lifetime=3600,
#                     )

#                     # First, connect without specifying database to discover
#                     # available databases
#                     _db_name = _get_database_name(driver)

#                     if not _db_name:
#                         logger.warning(
#                             "Could not detect database name, trying default 'neo4j'"
#                         )
#                         _db_name = "neo4j"

#                     # Test connection with the detected database
#                     with driver.session(database=_db_name) as session:
#                         result = session.run("RETURN 1 as test", timeout=10)
#                         result.single()

#                     logger.info(f"✅ Neo4j connected! Database:
# '{_db_name}' via {uri}")
#                     _driver = driver
#                     _neo4j_available = True
#                     return _driver

#                 except ServiceUnavailable as e:
#                     logger.warning(f"Failed with {uri}: {e}")
#                     if driver:
#                         driver.close()
#                     continue
#                 except Exception as e:
#                     logger.warning(f"Failed with {uri}: {e}")
#                     if driver:
#                         driver.close()
#                     continue

#             logger.error("All connection attempts failed")
#             _neo4j_available = False

#         except Exception as e:
#             logger.error(f"Neo4j driver creation failed: {e}")
#             _neo4j_available = False

#     return _driver


# # def _run_cypher(query: str, params: dict | None = None) -> list[dict[str, Any]]:
# #     """Execute a Cypher query using the auto-detected database name."""
# #     # global _db_name

# #     driver = _get_driver()
# #     if driver is None:
# #         return []

# #     try:
# #         # Use the detected database name
# #         with driver.session(database=_db_name) as session:
# #             result = session.run(query, params or {}, timeout=30)
# #             return [dict(record) for record in result]
# #     except Exception as e:
# #         logger.error(f"Cypher query failed: {e}")
# #         return []


# async def _run_cypher_async(
#     query: str, params: dict | None = None
# ) -> list[dict[str, Any]]:
#     """Async wrapper for Cypher execution using a thread to avoid blocking."""

#     # We offload the blocking driver.session call to a thread
#     def _execute():
#         driver = _get_driver()
#         if driver is None:
#             return []
#         try:
#             with driver.session(database=_db_name) as session:
#                 result = session.run(query, params or {}, timeout=30)
#                 return [dict(record) for record in result]
#         except Exception as e:
#             logger.error(f"Cypher query failed: {e}")
#             return []

#     return await asyncio.to_thread(_execute)


# def run_cypher_sync(query: str, params: dict | None = None) -> list[dict[str, Any]]:
#     """Synchronous version of query execution for UI use."""
#     return _run_sync(_run_cypher_async(query, params))


# def ensure_schema() -> None:
#     """Create uniqueness constraints if they don't exist yet."""

#     async def _internal_schema():
#         driver = _get_driver()
#         if driver is None:
#             logger.warning("Neo4j not available - skipping schema creation")
#             return

#         try:
#             constraints = [
#                 "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session)"
#                 "REQUIRE s.session_id IS UNIQUE",
#                 "CREATE CONSTRAINT agent_action_id_unique IF NOT EXISTS"
#                 "FOR (a:AgentAction)"
#                 "REQUIRE a.action_id IS UNIQUE",
#                 "CREATE CONSTRAINT mcp_call_id_unique IF NOT EXISTS"
#                 "FOR (m:MCPServerCall) "
#                 "REQUIRE m.call_id IS UNIQUE",
#             ]

#             for constraint in constraints:
#                 try:
#                     await _run_cypher_async(constraint)
#                     logger.debug("Constraint applied successfully")
#                 except Exception as e:
#                     logger.debug(f"Constraint may already exist: {e}")

#             logger.info(f"Neo4j schema ensured on database '{_db_name}'")
#         except Exception as exc:
#             logger.warning("Schema constraint creation issue: %s", exc)

#     _run_sync(_internal_schema())


# # @tools


# @tool
# def project_session_to_graph(session_json: str) -> str:
#     """
#     Project a full session execution trace into the Neo4j Aura DB
#     knowledge graph.

#     Extracts client-side AgentAction nodes and server-side MCPServerCall
#     nodes from the trace, then links them with typed directional edges:
#       (:Session)-[:TRIGGERED]->(:AgentAction)
#       (:AgentAction)-[:ROUTED_TO]->(:MCPServerCall)
#       (:MCPServerCall)-[:DEPENDS_ON]->(:AgentAction)

#     Args:
#         session_json: JSON string — the output of get_session_trace tool,
#                       containing a list of log entry dicts.

#     Returns:
#         JSON summary of nodes and edges written to Neo4j.
#     """
#     logger.info("project_session_to_graph called")
#     try:

#         async def task():
#             entries: list[dict[str, Any]] = json.loads(session_json)
#             if not entries:
#                 return json.dumps({"status": "no_entries", "nodes": 0, "edges": 0})

#             session_id = entries[0].get("session_id", str(uuid.uuid4()))
#             nodes_created = 0
#             edges_created = 0

#             # 1. Upsert Session node
#             query_count = sum(
#                 1
#                 for e in entries
#                 if e.get("component", "").startswith("agent.planning.query")
#             )
#             started_at = min((e.get("timestamp", 0.0) for e in entries), default=0.0)

#             await _run_cypher_async(
#                 """
#                 MERGE (s:Session {session_id: $session_id})
#                 ON CREATE SET s.started_at = $started_at,
#                             s.query_count = $query_count
#                 ON MATCH  SET s.query_count = $query_count
#                 """,
#                 {
#                     "session_id": session_id,
#                     "started_at": started_at,
#                     "query_count": query_count,
#                 },
#             )
#             nodes_created += 1

#             # 2. Process each log entry
#             prev_agent_action_id: str | None = None

#             for entry in entries:
#                 interaction_type = entry.get("mcp_interaction_type", "")
#                 component = entry.get("component", "")
#                 content = entry.get("content", "")[:500]
#                 timestamp = entry.get("timestamp", 0.0)
#                 metadata = entry.get("metadata", {})

#                 # AgentAction nodes
#                 if interaction_type in ("agent_action", "tool_observation"):
#                     action_id = f"{session_id}:{entry.get('key', str(uuid.uuid4()))}"
#                     await _run_cypher_async(
#                         """
#                         MERGE (a:AgentAction {action_id: $action_id})
#                         ON CREATE SET
#                             a.session_id            = $session_id,
#                             a.component             = $component,
#                             a.content               = $content,
#                             a.timestamp             = $timestamp,
#                             a.mcp_interaction_type  = $interaction_type
#                         """,
#                         {
#                             "action_id": action_id,
#                             "session_id": session_id,
#                             "component": component,
#                             "content": content,
#                             "timestamp": timestamp,
#                             "interaction_type": interaction_type,
#                         },
#                     )
#                     nodes_created += 1

#                     # Session -> AgentAction
#                     await _run_cypher_async(
#                         """
#                         MATCH (s:Session {session_id: $session_id})
#                         MATCH (a:AgentAction {action_id: $action_id})
#                         MERGE (s)-[:TRIGGERED]->(a)
#                         """,
#                         {"session_id": session_id, "action_id": action_id},
#                     )
#                     edges_created += 1
#                     prev_agent_action_id = action_id

#                 # MCPServerCall nodes
#                 elif interaction_type in (
#                     "tool_invocation",
#                     "resource_read",
#                     "sampling_request",
#                 ):
#                     call_id = f"{session_id}:{entry.get('key', str(uuid.uuid4()))}"
#                     tool_name = metadata.get("tool_name", component.split(".")[-1])
#                     latency_ms = int(metadata.get("latency_ms", 0))

#                     await _run_cypher_async(
#                         """
#                         MERGE (m:MCPServerCall {call_id: $call_id})
#                         ON CREATE SET
#                             m.session_id            = $session_id,
#                             m.tool_name             = $tool_name,
#                             m.latency_ms            = $latency_ms,
#                             m.timestamp             = $timestamp,
#                             m.content               = $content,
#                             m.mcp_interaction_type  = $interaction_type
#                         """,
#                         {
#                             "call_id": call_id,
#                             "session_id": session_id,
#                             "tool_name": tool_name,
#                             "latency_ms": latency_ms,
#                             "timestamp": timestamp,
#                             "content": content,
#                             "interaction_type": interaction_type,
#                         },
#                     )
#                     nodes_created += 1

#                     # AgentAction -> MCPServerCall  (ROUTED_TO)
#                     if prev_agent_action_id:
#                         await _run_cypher_async(
#                             """
#                             MATCH (a:AgentAction {action_id: $action_id})
#                             MATCH (m:MCPServerCall {call_id: $call_id})
#                             MERGE (a)-[:ROUTED_TO]->(m)
#                             """,
#                             {"action_id": prev_agent_action_id, "call_id": call_id},
#                         )
#                         edges_created += 1

#                     # MCPServerCall -> Session  (DEPENDS_ON)
#                     await _run_cypher_async(
#                         """
#                         MATCH (m:MCPServerCall {call_id: $call_id})
#                         MATCH (s:Session {session_id: $session_id})
#                         MERGE (m)-[:DEPENDS_ON]->(s)
#                         """,
#                         {"call_id": call_id, "session_id": session_id},
#                     )
#                     edges_created += 1

#             result = {
#                 "status": "success",
#                 "session_id": session_id,
#                 "nodes_created": nodes_created,
#                 "edges_created": edges_created,
#             }
#             logger.info("project_session_to_graph: %s", result)
#             return json.dumps(result, indent=2)

#         return _run_sync(task())
#     except Exception as exc:
#         logger.error("project_session_to_graph error: %s", exc)
#         return json.dumps({"status": "error", "message": str(exc)})


# @tool
# async def query_knowledge_graph(cypher_query: str) -> str:
#     """
#     Execute a read-only Cypher query against the Neo4j Aura DB graph
#     and return the results as JSON.

#     Useful for:
#       - Inspecting which sessions are stored in the graph.
#       - Tracing the path from a Session through AgentActions to
#         MCPServerCalls.
#       - Finding MCPServerCalls with high latency.
#       - Checking graph topology.

#     Example queries:
#       MATCH (s:Session) RETURN s.session_id, s.query_count LIMIT 10
#       MATCH (s:Session)-[:TRIGGERED]->(a:AgentAction) RETURN s,a LIMIT 5
#       MATCH (m:MCPServerCall) WHERE m.latency_ms > 2000 RETURN m

#     Args:
#         cypher_query: A valid Cypher read query string.

#     Returns:
#         JSON string with query results (list of row dicts).
#     """
#     logger.info("query_knowledge_graph | query='%s'", cypher_query[:120])
#     try:

#         async def task():
#             rows = await _run_cypher_async(cypher_query)
#             # Stringify non-serializable neo4j types
#             clean_rows = []
#             for row in rows:
#                 clean_row = {}
#                 for k, v in row.items():
#                     try:
#                         json.dumps(v)
#                         clean_row[k] = v
#                     except TypeError:
#                         clean_row[k] = str(v)
#                 clean_rows.append(clean_row)
#             logger.info("query_knowledge_graph returned %d rows", len(clean_rows))
#             return json.dumps(clean_rows, indent=2, default=str)

#         return _run_sync(task())
#     except Exception as exc:
#         logger.error("query_knowledge_graph error: %s", exc)
#         return json.dumps({"error": str(exc)})


# @tool
# async def get_graph_summary() -> str:
#     """
#     Return a high-level summary of the current state of the Neo4j
#     knowledge graph: node counts, edge counts, and the most recent
#     sessions.

#     No arguments required.

#     Returns:
#         JSON with node/edge counts and recent session list.
#     """
#     logger.info("get_graph_summary called")
#     try:

#         async def task():
#             counts, edges, recent_sessions = await asyncio.gather(
#                 _run_cypher_async("""
#                     MATCH (n)
#                     RETURN labels(n)[0] AS label, count(n) AS count
#                     ORDER BY count DESC
#                     """),
#                 _run_cypher_async("""
#                     MATCH ()-[r]->()
#                     RETURN type(r) AS relationship, count(r) AS count
#                     ORDER BY count DESC
#                     """),
#                 _run_cypher_async("""
#                     MATCH (s:Session)
#                     RETURN s.session_id AS session_id,
#                         s.query_count AS query_count,
#                         s.started_at  AS started_at
#                     ORDER BY s.started_at DESC
#                     LIMIT 5
#                     """),
#             )
#             # counts = _run_cypher("""
#             #     MATCH (n)
#             #     RETURN labels(n)[0] AS label, count(n) AS count
#             #     ORDER BY count DESC
#             #     """)
#             # edges = _run_cypher("""
#             #     MATCH ()-[r]->()
#             #     RETURN type(r) AS relationship, count(r) AS count
#             #     ORDER BY count DESC
#             #     """)
#             # recent_sessions = _run_cypher("""
#             #     MATCH (s:Session)
#             #     RETURN s.session_id AS session_id,
#             #            s.query_count AS query_count,
#             #            s.started_at  AS started_at
#             #     ORDER BY s.started_at DESC
#             #     LIMIT 5
#             #     """)
#             result = {
#                 "node_counts": counts,
#                 "edge_counts": edges,
#                 "recent_sessions": recent_sessions,
#             }
#             return json.dumps(result, indent=2, default=str)

#         return _run_sync(task())
#     except Exception as exc:
#         logger.error("get_graph_summary error: %s", exc)
#         return json.dumps({"error": str(exc)})
