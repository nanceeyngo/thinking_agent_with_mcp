"""
analysis_dashboard/app.py

Streamlit Diagnostic Dashboard - Human-in-the-loop control plane.

Features

  - Natural-language chat interface → Log Analysis Agent
  - Step-by-step agent reasoning display
  - Inline performance charts (matplotlib)
  - Neo4j graph update notifications
  - Recent log browser
  - Session selector for deep-dive traces
"""

from __future__ import annotations

import warnings
import base64
import json
import logging
import sys
import time
from pathlib import Path

import streamlit as st

# Suppress transformers import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*torchvision.*")

# Suppress streamlit file watcher warnings
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

logger = logging.getLogger("analysis_agent")

# Page config must be the very first Streamlit call
st.set_page_config(
    page_title="MCP Agent Diagnostics",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Logging
LOG_FILE = Path("analysis_agent.log")
_fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
_fh.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [DASHBOARD] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [DASHBOARD] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
ui_logger = logging.getLogger("dashboard")
if not ui_logger.handlers:
    ui_logger.addHandler(_fh)
    ui_logger.addHandler(_sh)
ui_logger.setLevel(logging.INFO)
ui_logger.propagate = False


# Session-state initialisation


def _init_state():
    defaults = {
        "messages": [],  # chat history
        "agent": None,  # cached agent instance
        "last_steps": [],  # steps from last agent run
        "last_charts": [],  # base64 PNG strings from last run
        "neo4j_updates": [],  # summary of graph writes
        "agent_ready": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# Agent bootstrap


@st.cache_resource(show_spinner="Initialising Log Analysis Agent…")
def _load_agent():
    """Load the analysis agent (cached across reruns)."""
    from analysis_dashboard.analysis_agent import build_analysis_agent
    from analysis_dashboard.neo4j_tools import ensure_schema

    try:
        ensure_schema()
        ui_logger.info("Neo4j schema ensured.")
    except Exception as exc:
        ui_logger.warning("Neo4j schema init warning: %s", exc)

    return build_analysis_agent()


def _get_agent():
    if st.session_state["agent"] is None:
        try:
            st.session_state["agent"] = _load_agent()
            st.session_state["agent_ready"] = True
        except Exception as exc:
            st.error(f"Failed to initialise agent: {exc}")
            ui_logger.error("Agent init failed: %s", exc)
    return st.session_state["agent"]


# Async helper


# def _run_async(coro):
#     """Run a coroutine from a sync Streamlit context."""
#     try:
#         loop = asyncio.get_event_loop()
#         if loop.is_running():
#             import concurrent.futures


#             with concurrent.futures.ThreadPoolExecutor() as pool:
#                 future = pool.submit(asyncio.run, coro)
#                 return future.result()
#         # else:
#         #     return loop.run_until_complete(coro)
#     except RuntimeError:
#         return asyncio.run(coro)
def _run_async(coro):
    """
    Run *coro* in the persistent background event loop owned by
    retrieval_tools._BackgroundLoop.

    Safe to call from Streamlit's sync script context on every rerun.
    """
    from analysis_dashboard.retrieval_tools import run_in_background

    return run_in_background(coro)


# Sidebar


def _render_sidebar():
    with st.sidebar:
        st.title("🔭 MCP Diagnostics")
        st.caption("Log Analysis Agent Control Plane")
        st.divider()

        # Connection status
        st.subheader("System Status")
        from analysis_dashboard.settings import settings

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Log DB",
                "Connected" if Path(settings.log_db_path).exists() else "Missing",
            )
        with col2:
            neo4j_ok = bool(settings.neo4j_uri)
            st.metric("Neo4j", "Configured" if neo4j_ok else "Not set")

        st.divider()

        # Quick actions
        st.subheader("Quick Diagnostics")
        if st.button("📊 Analyse Recent Logs", use_container_width=True):
            st.session_state["pending_query"] = (
                "List the most recent log entries and calculate latency trends. "
                "Generate a latency performance chart."
            )

        if st.button("🔍 Find Errors", use_container_width=True):
            st.session_state["pending_query"] = (
                "Search for any error or failure entries in the logs. "
                "Calculate error frequency and show me a chart."
            )

        if st.button("🗺️ Sync to Neo4j", use_container_width=True):
            st.session_state["pending_query"] = (
                "List the most recent logs, get a session trace for the most "
                "recent session_id you find, then project it to the Neo4j graph. "
                "Show me a graph summary after the sync."
            )

        if st.button("📈 Token Consumption", use_container_width=True):
            st.session_state["pending_query"] = (
                "Analyse token consumption patterns from sampling_request logs "
                "and generate a tokens chart."
            )

        st.divider()

        # Neo4j updates log
        if st.session_state["neo4j_updates"]:
            st.subheader("📝 Neo4j Updates This Session")
            for upd in st.session_state["neo4j_updates"][-5:]:
                st.info(upd, icon="✅")

        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["last_steps"] = []
            st.session_state["last_charts"] = []
            st.session_state["neo4j_updates"] = []
            st.rerun()


# Step renderer
def _render_steps(steps: list[dict]) -> list[str]:
    """
    Render agent reasoning steps in an expander.
    Returns list of base64-encoded chart images found in tool outputs.
    """
    from analysis_dashboard.analytics_tools import get_chart_image

    charts: list[str] = []
    if not steps:
        return charts

    with st.expander("🧠 Agent Reasoning Steps", expanded=False):
        for i, step in enumerate(steps, 1):
            step_type = step.get("type", "")

            if step_type == "tool_observation":
                tool_name = step.get("tool", "tool")
                content = step.get("content", "")

                # Show the Thought the agent had before calling this tool
                thought = step.get("thought", "")
                if thought:
                    thought_text = thought.split("\nAction:")[0].strip()
                    if thought_text:
                        st.markdown("**💭 Thought**")
                        st.markdown(f"> {thought_text}")

                st.markdown(f"**Step {i} — Tool:** `{tool_name}`")

                # Show tool input so you can see what was passed
                tool_input = step.get("tool_input", "")
                if tool_input:
                    with st.expander(f"📥 Input to `{tool_name}`", expanded=False):
                        st.code(str(tool_input)[:500], language="json")

                try:
                    parsed = json.loads(content)

                    # Fetch chart image from registry if this is a chart result
                    if isinstance(parsed, dict) and "chart_id" in parsed:
                        chart_id = parsed["chart_id"]
                        img_b64 = get_chart_image(chart_id)
                        if img_b64:
                            charts.append(img_b64)
                        st.json(parsed)

                    else:
                        st.json(parsed)

                    # Neo4j update notification
                    if isinstance(parsed, dict) and parsed.get("status") == "success":
                        if "nodes_created" in parsed:
                            msg = (
                                f"Neo4j updated — session: "
                                f"{parsed.get('session_id', '')[:12]}… | "
                                f"nodes: {parsed.get('nodes_created', 0)} | "
                                f"edges: {parsed.get('edges_created', 0)}"
                            )
                            st.session_state["neo4j_updates"].append(msg)
                            st.success(msg, icon="✅")

                except (json.JSONDecodeError, TypeError):
                    st.text_area(
                        "Output",
                        content[:1500],
                        height=100,
                        key=f"step_{i}_{tool_name}",
                    )
            elif step_type == "agent_thought":
                content = step.get("content", "")
                if content:
                    # Show thinking in a collapsible section to avoid clutter
                    with st.expander(f"💭 Step {i} — Agent Reasoning", expanded=False):
                        st.markdown(content[:1000])  # Show first 1000 chars of thinking
                        if len(content) > 1000:
                            st.caption(f"... ({len(content)} total characters)")

            st.divider()
            # elif step_type == "agent_thought":
            #     content = step.get("content", "")
            #     st.markdown(f"**Step {i} — Agent Thought**")
            #     st.markdown(f"> {content[:500]}")

            # st.divider()

    return charts


# def _render_steps(steps: list[dict]) -> list[str]:
#     """
#     Render agent reasoning steps in an expander.
#     Returns list of base64-encoded chart images found in tool outputs.
#     """
#     from analysis_dashboard.analytics_tools import get_chart_image

#     charts: list[str] = []
#     if not steps:
#         return charts

#     with st.expander("🧠 Agent Reasoning Steps", expanded=False):
#         for i, step in enumerate(steps, 1):
#             step_type = step.get("type", "")

#             if step_type == "tool_observation":
#                 tool_name = step.get("tool", "tool")
#                 content = step.get("content", "")

#                 st.markdown(f"**Step {i} — Tool:** `{tool_name}`")

#                 try:
#                     parsed = json.loads(content)

#                     # Fetch chart image from registry if this is a chart result
#                     if isinstance(parsed, dict) and "chart_id" in parsed:
#                         chart_id = parsed["chart_id"]
#                         img_b64 = get_chart_image(chart_id)
#                         if img_b64:
#                             charts.append(img_b64)
#                         # Show the summary without any blob
#                         st.json(parsed)

#                     else:
#                         st.json(parsed)

#                     # Neo4j update notification
#                     if isinstance(parsed, dict) and parsed.get("status") == "success":
#                         if "nodes_created" in parsed:
#                             msg = (
#                                 f"Neo4j updated — session: "
#                                 f"{parsed.get('session_id', '')[:12]}… | "
#                                 f"nodes: {parsed.get('nodes_created', 0)} | "
#                                 f"edges: {parsed.get('edges_created', 0)}"
#                             )
#                             st.session_state["neo4j_updates"].append(msg)
#                             st.success(msg, icon="✅")

#                 except (json.JSONDecodeError, TypeError):
#                     st.text_area(
#                         "Output",
#                         content[:1500],
#                         height=100,
#                         key=f"step_{i}_{tool_name}",
#                     )

#             elif step_type == "agent_thought":
#                 content = step.get("content", "")
#                 st.markdown(f"**Step {i} — Agent Thought**")
#                 st.markdown(f"> {content[:500]}")

#             st.divider()

#     return charts


# def _render_steps(steps: list[dict]) -> list[str]:
#     """
#     Render agent reasoning steps in an expander.
#     Returns list of base64-encoded chart images found in tool outputs.
#     """
#     from analysis_dashboard.analytics_tools import get_chart_image

#     charts: list[str] = []
#     if not steps:
#         return charts

#     with st.expander("🧠 Agent Reasoning Steps", expanded=False):
#         for i, step in enumerate(steps, 1):
#             step_type = step.get("type", "")

#             if step_type == "tool_observation":
#                 tool_name = step.get("tool", "tool")
#                 content = step.get("content", "")

#                 st.markdown(f"**Step {i} — Tool:** `{tool_name}`")

#                 # Try to parse and pretty-print JSON tool outputs
#                 try:
#                     parsed = json.loads(content)

#                     # Extract chart image if present
#                     if isinstance(parsed, dict) and "image_base64" in parsed:
#                         charts.append(parsed["image_base64"])
#                         # Show graph summary without the raw b64 blob
#                         display = {
#                             k: v for k, v in parsed.items() if k != "image_base64"
#                         }
#                         st.json(display)
#                     else:
#                         st.json(parsed)

#                     # Neo4j update notification
#                     if isinstance(parsed, dict) and parsed.get("status") == "success":
#                         if "nodes_created" in parsed:
#                             msg = (
#                                 f"Neo4j updated — session: "
#                                 f"{parsed.get('session_id', '')[:12]}… | "
#                                 f"nodes: {parsed.get('nodes_created', 0)} | "
#                                 f"edges: {parsed.get('edges_created', 0)}"
#                             )
#                             st.session_state["neo4j_updates"].append(msg)
#                             st.success(msg, icon="✅")

#                 except (json.JSONDecodeError, TypeError):
#                     st.text_area(
#                         "Output",
#                         content[:1500],
#                         height=100,
#                         key=f"step_{i}_{tool_name}",
#                     )

#             elif step_type == "agent_thought":
#                 content = step.get("content", "")
#                 st.markdown(f"**Step {i} — Agent Thought**")
#                 st.markdown(f"> {content[:500]}")

#             st.divider()

#     return charts


# Chart renderer


def _render_charts(charts: list[str]) -> None:
    if not charts:
        return
    st.subheader("📊 Performance Charts")
    cols = st.columns(min(len(charts), 2))
    for i, b64 in enumerate(charts):
        with cols[i % 2]:
            try:
                img_bytes = base64.b64decode(b64)
                st.image(img_bytes, use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not render chart: {exc}")


# Main chat panel


def _render_chat():
    st.title("🔭 MCP Agent Diagnostic Dashboard")
    st.caption(
        "Ask the Log Analysis Agent about system health, "
        "execution traces, latency trends, or Neo4j graph status."
    )

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("charts"):
                _render_charts(msg["charts"])

    # Handle quick-action buttons
    if "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")
        _handle_query(query)
        st.rerun()

    # Chat input
    if prompt := st.chat_input(
        "Ask about logs, errors, latency, or graph state…",
        key="chat_input",
    ):
        _handle_query(prompt)


def _handle_query(query: str) -> None:
    """Process a user query through the analysis agent."""
    agent = _get_agent()
    if agent is None:
        st.error("Agent not available. Check your API keys in .env.")
        return

    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            ui_logger.info("Dashboard query: %s", query[:200])
            t0 = time.perf_counter()

            try:
                from analysis_dashboard.analysis_agent import run_analysis_query

                answer, steps = _run_async(run_analysis_query(agent, query))
                elapsed = time.perf_counter() - t0
                ui_logger.info("Query completed in %.1fs", elapsed)
            except Exception as exc:
                answer = f"Agent error: {exc}"
                steps = []
                ui_logger.error("Query error: %s", exc)

        # Render reasoning steps and collect charts
        charts = _render_steps(steps)
        _render_charts(charts)

        # Display final answer
        if answer:
            st.markdown(answer)
        else:
            st.info(
                "The agent completed its analysis — check the reasoning steps above."
            )

        # Store in history
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer or "Analysis complete — see reasoning steps.",
                "charts": charts,
            }
        )
        st.session_state["last_steps"] = steps
        st.session_state["last_charts"] = charts


# Log browser tab


def _render_log_browser():
    st.header("📋 Log Browser")
    st.caption("Browse raw log entries from the vector store.")

    col1, col2 = st.columns([2, 1])
    with col1:
        ns_input = st.text_input(
            "Namespace prefix",
            value="logs",
            help="Dot-separated, e.g. 'logs.mcp.client'",
        )
    with col2:
        limit = st.slider("Max entries", 5, 100, 30)

    if st.button("Load Logs"):
        with st.spinner("Loading…"):
            try:
                # Use the same sync store as the tools
                from analysis_dashboard.retrieval_tools import search_sync

                # store = _get_store()
                ns_tuple = tuple(ns_input.split(".")) if ns_input else ("logs",)
                results = search_sync(ns_tuple, "", limit)
                # results = _search_sync(ns_tuple, "", limit)
                # results = store.search(ns_tuple, query="", limit=limit)

                if results:
                    rows = []
                    for item in results:
                        v = item.value if hasattr(item, "value") else {}
                        if isinstance(v, dict):
                            rows.append(
                                {
                                    "session_id": str(v.get("session_id", ""))[:16]
                                    + "…",
                                    "type": v.get("mcp_interaction_type", ""),
                                    "component": v.get("component", ""),
                                    "content_snippet": str(v.get("content", ""))[:80]
                                    + "…",
                                    "timestamp": v.get("timestamp", 0.0),
                                }
                            )

                    if rows:
                        import pandas as pd

                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True)
                        st.caption(f"{len(rows)} entries loaded.")
                    else:
                        st.info("No entries could be parsed from results.")
                else:
                    st.info("No entries found in this namespace.")
            except Exception as exc:
                st.error(f"Could not load logs: {exc}")
                logger.error(f"Log browser error: {exc}", exc_info=True)


# def _render_log_browser():
#     st.header("📋 Log Browser")
#     st.caption("Browse raw log entries from the vector store.")

#     col1, col2 = st.columns([2, 1])
#     with col1:
#         ns_input = st.text_input(
#             "Namespace prefix",
#             value="logs",
#             help="Dot-separated, e.g. 'logs.mcp.client'",
#         )
#     with col2:
#         limit = st.slider("Max entries", 5, 100, 30)

#     if st.button("Load Logs"):
#         with st.spinner("Loading…"):
#             try:
# from analysis_dashboard.retrieval_tools import _get_store_and_embeddings

#                 store, _ = _get_store_and_embeddings()
#                 ns_tuple = tuple(ns_input.split(".")) if ns_input else ("logs",)
#                 results = store.search(
# namespace_prefix=ns_tuple, query="", limit=limit)
#                 if results:
#                     rows = []
#                     for item in results:
#                         v = item.value
#                         rows.append(
#                             {
#                                 "session_id": v.get("session_id", "")[:16] + "…",
#                                 "type": v.get("mcp_interaction_type", ""),
#                                 "component": v.get("component", ""),
#                                 "content_snippet": v.get("content", "")[:80] + "…",
#                                 "timestamp": v.get("timestamp", 0.0),
#                             }
#                         )
#                     import pandas as pd

#                     df = pd.DataFrame(rows)
#                     st.dataframe(df, use_container_width=True)
#                     st.caption(f"{len(rows)} entries loaded.")
#                 else:
#                     st.info("No entries found in this namespace.")
#             except Exception as exc:
#                 st.error(f"Could not load logs: {exc}")


# Graph explorer tab


def _render_graph_explorer():
    st.header("🗺️ Neo4j Graph Explorer")

    if st.button("Refresh Graph Summary"):
        with st.spinner("Querying Neo4j…"):
            try:
                from analysis_dashboard.neo4j_tools import run_cypher_sync, _get_driver

                driver = _get_driver()
                if driver is None:
                    st.warning("Neo4j is not connected. Check your credentials in .env")
                    return

                counts = run_cypher_sync(
                    "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count "
                    "ORDER BY count DESC"
                )
                edges = run_cypher_sync(
                    "MATCH ()-[r]->() RETURN type(r) AS relationship, "
                    "count(r) AS count ORDER BY count DESC"
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Node Counts")
                    if counts:
                        import pandas as pd

                        st.dataframe(pd.DataFrame(counts), use_container_width=True)
                    else:
                        st.info(
                            "No nodes found. Run the client first to generate"
                            "logs, then sync to Neo4j."
                        )

                with col2:
                    st.subheader("Edge Counts")
                    if edges:
                        import pandas as pd

                        st.dataframe(pd.DataFrame(edges), use_container_width=True)
                    else:
                        st.info(
                            "No edges found. Sync a session"
                            "to Neo4j to create relationships."
                        )

            except Exception as exc:
                st.error(f"Neo4j query error: {exc}")

    st.divider()
    st.subheader("Custom Cypher Query")
    cypher = st.text_area(
        "Cypher",
        value="MATCH (s:Session) RETURN s.session_id, s.query_count LIMIT 10",
        height=80,
    )
    if st.button("Run Cypher"):
        with st.spinner("Running…"):
            try:
                from analysis_dashboard.neo4j_tools import (
                    run_cypher_sync,
                    _get_driver,
                )

                rows = run_cypher_sync(cypher)
                if rows:
                    import pandas as pd

                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info(
                        "Query returned no results. No data has"
                        "been synced to Neo4j yet."
                    )
            except Exception as exc:
                st.error(f"Cypher error: {exc}")


# def _render_graph_explorer():
#     st.header("🗺️ Neo4j Graph Explorer")

#     if st.button("Refresh Graph Summary"):
#         with st.spinner("Querying Neo4j…"):
#             try:
#                 from analysis_dashboard.neo4j_tools import _run_cypher

#                 counts = _run_cypher(
#                     "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count "
#                     "ORDER BY count DESC"
#                 )
#                 edges = _run_cypher(
#                     "MATCH ()-[r]->() RETURN type(r) AS relationship, "
#                     "count(r) AS count ORDER BY count DESC"
#                 )

#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.subheader("Node Counts")
#                     if counts:
#                         import pandas as pd

#                         st.dataframe(pd.DataFrame(counts), use_container_width=True)
#                     else:
#                         st.info("No nodes found.")
#                 with col2:
#                     st.subheader("Edge Counts")
#                     if edges:
#                         import pandas as pd

#                         st.dataframe(pd.DataFrame(edges), use_container_width=True)
#                     else:
#                         st.info("No edges found.")

#             except Exception as exc:
#                 st.error(f"Neo4j query error: {exc}")

#     st.divider()
#     st.subheader("Custom Cypher Query")
#     cypher = st.text_area(
#         "Cypher",
#         value="MATCH (s:Session) RETURN s.session_id, s.query_count LIMIT 10",
#         height=80,
#     )
#     if st.button("Run Cypher"):
#         with st.spinner("Running…"):
#             try:
#                 from analysis_dashboard.neo4j_tools import _run_cypher

#                 rows = _run_cypher(cypher)
#                 if rows:
#                     import pandas as pd

#                     st.dataframe(pd.DataFrame(rows), use_container_width=True)
#                 else:
#                     st.info("Query returned no results.")
#             except Exception as exc:
#                 st.error(f"Cypher error: {exc}")


# Entry point


def main():
    """Main Streamlit app entry point."""
    _render_sidebar()

    tab_chat, tab_logs, tab_graph = st.tabs(
        [
            "💬 Diagnostic Chat",
            "📋 Log Browser",
            "🗺️ Graph Explorer",
        ]
    )

    with tab_chat:
        _render_chat()

    with tab_logs:
        _render_log_browser()

    with tab_graph:
        _render_graph_explorer()


if __name__ == "__main__":
    main()
