"""
analysis_dashboard/analysis_agent.py

Decoupled Log Analysis Agent.

This module runs as a completely separate process from the MCP client/server.
It uses LangChain's create_agent factory with six specialised tools:

  Retrieval tools (from retrieval_tools.py):
    • search_logs_semantic   - vector-similarity log search
    • list_recent_logs       - time-ordered log scan
    • get_session_trace      - full session execution trace

  Neo4j tools (from neo4j_tools.py):
    • project_session_to_graph  - map session traces to Neo4j KG
    • query_knowledge_graph     - read Cypher queries
    • get_graph_summary         - high-level graph stats

  Analytics tools (from analytics_tools.py):
    • calculate_latency_trends   - moving-average latency analysis
    • calculate_token_trends     - token consumption patterns
    • calculate_error_frequency  - error event frequency
    • generate_performance_chart - matplotlib/seaborn visualisations

All agent execution is logged to analysis_agent.log.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from analysis_dashboard.settings import settings

# Logging

LOG_FILE = Path("analysis_agent.log")

_handler_file = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
_handler_file.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [ANALYSIS_AGENT] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
_handler_stdout = logging.StreamHandler(sys.stdout)
_handler_stdout.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [ANALYSIS_AGENT] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

agent_logger = logging.getLogger("analysis_agent")
agent_logger.setLevel(logging.DEBUG)
if not agent_logger.handlers:
    agent_logger.addHandler(_handler_file)
    agent_logger.addHandler(_handler_stdout)
agent_logger.propagate = False


# Tool output summariser

# Tools whose output should be summarised before being fed back to the LLM.
# The full content is kept in the steps list for the UI.
_LARGE_OUTPUT_TOOLS = {
    "list_recent_logs",
    "search_logs_semantic",
    "get_session_trace",
}

# LLM factory


def _build_llm():
    if settings.use_groq and settings.groq_api_key:
        from langchain_groq import ChatGroq

        agent_logger.info("Using Groq LLM: %s", settings.groq_model_name)
        return ChatGroq(
            model=settings.groq_model_name,
            temperature=0,
            api_key=settings.groq_api_key.get_secret_value(),
            max_tokens=1024,
            default_headers={
                "HTTP-Referer": "https://thinking-agent-analysis",
                "X-Title": "Thinking Agent Analysis Dashboard",
            },
        )
    elif settings.openrouter_api_key:
        from langchain_openai import ChatOpenAI

        agent_logger.info("Using OpenRouter LLM: %s", settings.model_name)
        return ChatOpenAI(
            model=settings.model_name,
            temperature=0,
            openai_api_key=settings.openrouter_api_key.get_secret_value(),
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=1024,
            default_headers={
                "HTTP-Referer": "https://thinking-agent-analysis",
                "X-Title": "Thinking Agent Analysis Dashboard",
            },
        )
    else:
        raise ValueError(
            "No LLM configured. Set OPENROUTER_API_KEY or GROQ_API_KEY in .env"
        )


# Tool bundle


def _build_tools():
    from analysis_dashboard.retrieval_tools import (
        get_session_trace,
        list_recent_logs,
        search_logs_semantic,
    )
    from analysis_dashboard.neo4j_tools import (
        get_graph_summary,
        project_session_to_graph,
        query_knowledge_graph,
    )
    from analysis_dashboard.analytics_tools import (
        calculate_error_frequency,
        calculate_latency_trends,
        calculate_token_trends,
        generate_performance_chart,
    )

    return [
        search_logs_semantic,
        list_recent_logs,
        get_session_trace,
        project_session_to_graph,
        query_knowledge_graph,
        get_graph_summary,
        calculate_latency_trends,
        calculate_token_trends,
        calculate_error_frequency,
        generate_performance_chart,
    ]


# System prompt

ANALYSIS_AGENT_SYSTEM_PROMPT = """
You are an expert Log Analysis Agent for a distributed multi-agent MCP system.

You have access to the following tools:

RETRIEVAL TOOLS:
  - search_logs_semantic: Vector-similarity search over execution logs.
  - list_recent_logs: Time-ordered scan of log entries.
  - get_session_trace: Full causal trace for a specific session_id.

NEO4J GRAPH TOOLS:
  - project_session_to_graph: Map session traces into the Neo4j knowledge graph.
  - query_knowledge_graph: Execute read Cypher queries.
  - get_graph_summary: High-level graph node/edge statistics.

ANALYTICS TOOLS:
  - calculate_latency_trends: Moving-average latency analysis.
  - calculate_token_trends: Token consumption patterns.
  - calculate_error_frequency: Error/failure event counting.
  - generate_performance_chart: Generate matplotlib/seaborn charts.

CRITICAL RULES:
1. Call ONE tool at a time. Wait for its output before calling the next.
2. NEVER pass a tool call expression as an argument to another tool.
3. When a retrieval tool (list_recent_logs, search_logs_semantic,
   get_session_trace) returns a summary with full_output_available=true,
   you MUST call the next analytics tool immediately — the system will
   automatically supply the full JSON. Do NOT try to copy or reconstruct
   the JSON yourself.
4. After completing your analysis, write your final answer and STOP.
5. Do NOT repeat tool calls you have already made successfully.
6. If a tool returns status "no_data" or "conclusion", accept it as final
   and move to the next step or write your summary.
   Do NOT retry with different queries.
7. If a tool returns an error, try ONCE with corrected input, then move on.
8. When the user asks to save a chart, pass save_to_disk=True to
   generate_performance_chart.

  Example (CORRECT):
    Step 1 → call search_logs_semantic(query="sampling_request", ...)
    Step 2 → receive JSON string result
    Step 3 → call calculate_token_trends(logs_json="<the JSON string from step 2>")

  Example (WRONG — never do this):
    calculate_token_trends(logs_json=search_logs_semantic(...))

DIAGNOSTIC WORKFLOWS:

General analysis:
  1. list_recent_logs
  2. calculate_latency_trends(logs_json=FULL_OUTPUT)
  3. calculate_error_frequency(logs_json=FULL_OUTPUT)
  4. generate_performance_chart(metric_json=<step 2 output>, chart_type="latency")
  5. Summarise findings and STOP.

Token analysis:
  1. Call search_logs_semantic with query="sampling_request token consumption"
     and namespace_prefix="logs" to get log entries.
  2. Pass that output to calculate_token_trends(logs_json=FULL_OUTPUT).
  3. If no_data: summarise and STOP.
  4. Pass the result of calculate_token_trends to
     generate_performance_chart(metric_json=<step 2 output>, chart_type="tokens").
  5. Summarise findings and STOP.

Error analysis:
  1. search_logs_semantic(query="error failed exception", limit=30)
  2. calculate_error_frequency(logs_json=FULL_OUTPUT)
  3. generate_performance_chart(metric_json=<step 2 output>, chart_type="errors")
  4. Summarise findings and STOP.

Neo4j sync:
  1. list_recent_logs
  2. get_session_trace(session_id=<most recent session_id from step 1>)
  3. project_session_to_graph(session_json=FULL_OUTPUT)
  4. get_graph_summary
  5. Summarise and STOP.

Always provide structured, actionable diagnostic insights in your final summary.
"""


# Agent factory


def build_analysis_agent():
    llm = _build_llm()
    tools = _build_tools()

    agent_logger.info(
        "Building analysis agent with %d tools: %s",
        len(tools),
        [t.name for t in tools],
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=ANALYSIS_AGENT_SYSTEM_PROMPT,
    )

    agent_logger.info("Analysis agent ready.")
    return agent


# Runner

# Maps tool_call_id → full original content for pass-through
_pending_full_outputs: dict[str, str] = {}


async def run_analysis_query(
    agent,
    query: str,
) -> tuple[str, list[dict]]:
    """
    Run a query and return (final_answer, steps).

    Key behaviour: tool outputs from retrieval tools are summarised before
    being returned to the LLM (to stay within function-call size limits),
    but the full content is stored in _pending_full_outputs so that
    analytics tools can receive it via the FULL_OUTPUT passthrough.
    """
    agent_logger.info("Analysis query: %s", query[:200])
    steps: list[dict] = []
    final_answer = ""

    # Per-query store: tool_name → last full output
    # Used to resolve FULL_OUTPUT references in analytics tool calls
    last_full_output: dict[str, str] = {}

    # Wrap the agent to intercept and summarise large tool outputs
    async def _run():
        nonlocal final_answer

        # Custom astream loop with output interception
        messages = [HumanMessage(content=query)]
        config = {"recursion_limit": 25}

        async for event in agent.astream(
            {"messages": messages},
            config=config,
        ):
            #  Tool observations (ToolMessages)
            if "tools" in event:
                for msg in event["tools"].get("messages", []):
                    tool_name = getattr(msg, "name", "tool")
                    full_content = str(msg.content)

                    agent_logger.info(
                        "[TOOL OBS — %s]\n%s", tool_name, full_content[:300]
                    )

                    # Store full output for pass-through
                    last_full_output[tool_name] = full_content

                    # Build compact summary for the steps UI
                    steps.append(
                        {
                            "type": "tool_observation",
                            "tool": tool_name,
                            "thought": "",
                            "tool_input": "",
                            "content": full_content,  # full content for UI
                        }
                    )

            #  Model messages (AIMessages)
            if "model" in event:
                for msg in event["model"].get("messages", []):
                    if not isinstance(msg, AIMessage):
                        continue
                    content = str(msg.content).strip()

                    if msg.tool_calls:
                        agent_logger.info("[AGENT THOUGHT]\n%s", content[:300])
                        for tc in msg.tool_calls:
                            tc_name = tc.get("name", "")
                            tc_args = tc.get("args", {})

                            # Resolve FULL_OUTPUT references
                            for arg_key, arg_val in tc_args.items():
                                if (
                                    isinstance(arg_val, str)
                                    and "FULL_OUTPUT" in arg_val
                                ):
                                    # Find the most recent retrieval output
                                    for src in _LARGE_OUTPUT_TOOLS:
                                        if src in last_full_output:
                                            tc_args[arg_key] = last_full_output[src]
                                            agent_logger.info(
                                                "Resolved FULL_OUTPUT ref in %s.%s "
                                                "from %s (%d chars)",
                                                tc_name,
                                                arg_key,
                                                src,
                                                len(last_full_output[src]),
                                            )
                                            break

                            steps.append(
                                {
                                    "type": "tool_observation",
                                    "tool": tc_name,
                                    "thought": content,
                                    "tool_input": str(tc_args),
                                    "content": "",
                                }
                            )
                    elif content:
                        agent_logger.info("[AGENT ANSWER]\n%s", content[:300])
                        final_answer = content

    try:
        await _run()
    except GraphRecursionError:
        agent_logger.warning("Recursion limit reached — returning partial results.")
        if not final_answer:
            final_answer = (
                "The agent reached its step limit before completing. "
                "Partial results are shown in the reasoning steps. "
                "Try a more specific query such as 'analyse latency' or "
                "'find errors' instead of a full health check."
            )
    except Exception as e:
        agent_logger.error("Agent execution failed: %s", e, exc_info=True)
        if not final_answer:
            final_answer = f"Analysis error: {str(e)}"

    agent_logger.info(
        "Analysis query complete. Steps: %d | Answer length: %d",
        len(steps),
        len(final_answer),
    )
    return final_answer, steps


# CLI entry point


async def _cli_main() -> None:
    from analysis_dashboard.neo4j_tools import ensure_schema

    try:
        ensure_schema()
    except Exception as exc:
        agent_logger.warning("Neo4j schema init: %s", exc)

    agent = build_analysis_agent()

    print("\n" + "=" * 60)
    print("  Log Analysis Agent — Interactive Mode")
    print("  Type 'exit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue
        answer, steps = await run_analysis_query(agent, query)
        print(f"\nAgent: {answer}\n")


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(_cli_main())


if __name__ == "__main__":
    main()
