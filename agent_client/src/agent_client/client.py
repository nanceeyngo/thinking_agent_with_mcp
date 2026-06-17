"""
agent_client/client.py

MCP Client + LangChain AI Agent (Stage 3 - Vector Log Store).

Responsibilities:
  1. Connect to the FastMCP server over streamable-http
  2. Wrap remote MCP tools as local LangChain @tool functions
  3. Handle incoming MCP Sampling requests (execute LLM calls on behalf
     of server)
  4. Initialize LangChain create_agent with wrapped tools
  5. Write dual-stream logs ([CLIENT] / [SERVER]) to agent_system.log

New in Stage 3:
- Every significant event is persisted to a LangGraph SQLite vector
  store (log_store.py) under a hierarchical dot-separated namespace.
- The flat agent_system.log file is retained for human readability.
- LogEntry objects carry a validated schema:
    session_id, mcp_interaction_type, component, content, metadata.
- Integer dict-keys are re-cast after JSON round-trips.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastmcp import Client
from fastmcp.client.logging import LogMessage
from fastmcp.client.sampling import SamplingMessage, SamplingParams
from langchain.agents import create_agent
from langchain.tools import tool
from mcp.shared.context import RequestContext
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_client.logging_config import (
    get_server_log_writer,
    setup_client_logger,
)

from agent_client.log_store import LogEntry, write_log

# 1. Settings


class Settings(BaseSettings):
    openrouter_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    use_groq: bool = False
    groq_model_name: str = "llama-3.3-70b-versatile"
    # Free-tier model that doesn't require credit
    model_name: str = "nvidia/nemotron-3-super-120b-a12b:free"  # noqa
    model_temperature: float = 0.0
    mcp_server_url: str = "http://localhost:8080/mcp"
    initialization_timeout: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # type: ignore[call-arg]

# 2. Loggers


client_log = setup_client_logger("agent_client")
server_log = get_server_log_writer()

# 3. Session tracking

# A single SESSION_ID identifies all log entries in this process run.
# This UUID is stored in every LogEntry so the analysis agent can
# reconstruct the full execution trace for a session.

SESSION_ID: str = str(uuid.uuid4())
client_log.info("Session started | session_id=%s", SESSION_ID)


# 4. LLM (used locally for sampling AND for the agent)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# llm = ChatOpenAI(
#     model=settings.model_name,
#     temperature=settings.model_temperature,
#     openai_api_key=settings.openrouter_api_key.get_secret_value(),
#     openai_api_base=OPENROUTER_BASE_URL,
#     max_tokens=2048,
#     default_headers={
#         "HTTP-Referer": "https://thinking-agent-stage3",
#         "X-Title": "Thinking Agent Stage 3",
#     },
# )


def _get_llm():
    if settings.use_groq and settings.groq_api_key:
        from langchain_groq import ChatGroq  # type: ignore[import]

        client_log.info("Using Groq LLM: %s", settings.groq_model_name)
        return ChatGroq(
            model=settings.groq_model_name,
            temperature=settings.model_temperature,
            api_key=settings.groq_api_key.get_secret_value(),
            max_tokens=2048,
            default_headers={
                "HTTP-Referer": "https://thinking-agent-analysis",
                "X-Title": "Thinking Agent Analysis Dashboard",
            },
        )
    elif settings.openrouter_api_key:
        from langchain_openai import ChatOpenAI

        client_log.info("Using OpenRouter LLM: %s", settings.model_name)
        return ChatOpenAI(
            model=settings.model_name,
            temperature=settings.model_temperature,
            openai_api_key=settings.openrouter_api_key.get_secret_value(),
            openai_api_base=OPENROUTER_BASE_URL,
            max_tokens=2048,
            default_headers={
                "HTTP-Referer": "https://thinking-agent-analysis",
                "X-Title": "Thinking Agent Analysis Dashboard",
            },
        )
    else:
        raise ValueError(
            "No LLM configured. Set OPENROUTER_API_KEY or GROQ_API_KEY in .env"
        )


llm = _get_llm()

client_log.info("LLM initialised")

# 5. Helpers

LOGGING_LEVEL_MAP = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}
_store_warn_count = 0  # rate-limit repeated store warnings


async def _store_entry(
    mcp_interaction_type: str,
    component: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Write a single log trace to the LangGraph vector store.

    On first failure: logs a WARNING (visible to operator).
    Repeated failures: suppressed after 3 to avoid log spam.
    The flat agent_system.log is always written regardless.
    """

    global _store_warn_count
    try:
        entry = LogEntry(
            session_id=SESSION_ID,
            mcp_interaction_type=mcp_interaction_type,
            component=component,
            content=content,
            metadata=metadata or {},
        )
        result = await write_log(entry)
        # If result is empty string, write failed silently
        if not result and _store_warn_count == 0:
            client_log.warning(
                "Vector store write returned empty key. " "Check database connection."
            )
        # if result.get("store_type") == "memory" and _store_warn_count == 0:
        #     client_log.warning(
        #         "Vector store is using InMemoryStore (not SQLite). "
        #         "Logs will NOT persist. Check SqliteStore install."
        #     )
    except Exception as exc:
        _store_warn_count += 1
        if _store_warn_count <= 3:
            client_log.warning(
                "Vector store write failed (attempt %d): %s — ",
                _store_warn_count,
                exc,
            )
        elif _store_warn_count == 4:
            client_log.warning(
                "Vector store write errors suppressed after 3 attempts. "
                "Flat log (agent_system.log) continues normally."
            )
    #     write_log(entry)
    # except Exception as exc:
    #     client_log.debug("Vector store write skipped: %s", exc)


# 6. MCP Callback Handlers


async def log_handler(message: LogMessage) -> None:
    """
    Receive MCP log notifications forwarded from the server; persist to vector store.
    """
    level_name = message.level.lower() if isinstance(message.level, str) else "info"
    level = LOGGING_LEVEL_MAP.get(level_name, 20)

    # Extract message text - FastMCP sends data as dict or string
    if isinstance(message.data, dict):
        msg_text = (
            message.data.get("msg") or message.data.get("message") or str(message.data)
        )
    else:
        msg_text = str(message.data)

    server_log.log(level, msg_text)

    # Persist server notification to vector store
    await _store_entry(
        mcp_interaction_type="tool_observation",
        component="mcp.server.notification",
        content=msg_text,
        metadata={"level": level_name, "session_id": SESSION_ID},
    )


async def sampling_handler(
    messages: list[SamplingMessage],
    params: SamplingParams,
    context: RequestContext,
) -> str:
    """
    Handle MCP Sampling requests from the server.
    Logs both the request and the response to the vector store.
    """
    t0 = time.perf_counter()
    client_log.info(
        "MCP Sampling request received from server | request_id=%s", context.request_id
    )

    # Import LangChain message types
    from langchain_core.messages import HumanMessage, SystemMessage

    # Properly convert SamplingMessages to LangChain messages
    lc_messages: list = []
    system_message: str | None = None

    for msg in messages:
        # Handle both SamplingMessage objects and dict representations
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content_obj = msg.get("content", {})
            if isinstance(content_obj, dict):
                text_content = content_obj.get("text", "")
            else:
                text_content = str(content_obj)
        else:
            # SamplingMessage object from mcp.types
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", None)
            if content is None:
                text_content = ""
            elif isinstance(content, dict):
                # TextContent as dict: {"type": "text", "text": "..."}
                text_content = content.get("text", "")
            elif hasattr(content, "text"):
                # TextContent object with .text attribute
                text_content = content.text
            else:
                text_content = str(content)

        if role == "system":
            system_message = text_content

        elif role == "assistant" and system_message is None:
            system_message = text_content

        elif role == "user":
            lc_messages.append(HumanMessage(content=text_content))

    # Log the sampling request to vector store
    request_content = (
        system_message
        or (lc_messages[0].content if lc_messages else "")
        or "sampling_request"
    )

    await _store_entry(
        mcp_interaction_type="sampling_request",
        component="mcp.sampling.request",
        content=request_content[:2000],
        metadata={
            "request_id": str(context.request_id),
            "message_count": len(messages),
            "max_tokens": getattr(params, "maxTokens", None),
        },
    )

    # If no system message in messages, but we have text to process,
    # treat the first user message as the main prompt
    # (system message if provided should take precedence)

    # client_log.debug(
    #     "Sampling request converted | messages=%d | system_msg=%s | "
    #     "max_tokens=%s | temperature=%s",
    #     len(messages),
    #     "present" if system_message else "absent",
    #     getattr(params, "maxTokens", "default"),
    #     getattr(params, "temperature", "default"),
    # )

    try:
        # Build message list with system message if present
        final_messages = []
        if system_message:
            final_messages.append(SystemMessage(content=system_message))
        final_messages.extend(lc_messages)

        # Extract sampling parameters
        max_tokens = getattr(params, "maxTokens", None)
        temperature = getattr(params, "temperature", None)

        # Call LLM with proper parameters
        response = await llm.ainvoke(
            final_messages,
            config={
                "max_tokens": max_tokens or 2048,
                "temperature": temperature if temperature is not None else 0.0,
            },
        )
        result_text = response.content
        latency_ms = int((time.perf_counter() - t0) * 1000)
        client_log.info(
            "Sampling response generated | length=%d chars", len(result_text)
        )

        # Log sampling response
        await _store_entry(
            mcp_interaction_type="sampling_request",
            component="mcp.sampling.response",
            content=str(result_text)[:2000],
            metadata={
                "request_id": str(context.request_id),
                "latency_ms": latency_ms,
                "response_length": len(str(result_text)),
            },
        )
        return result_text

    except Exception as exc:
        client_log.error("Sampling LLM call failed: %s", exc)
        await _store_entry(
            mcp_interaction_type="sampling_request",
            component="mcp.sampling.error",
            content=f"Sampling error: {exc}",
            metadata={"error": str(exc), "request_id": str(context.request_id)},
        )
        return f"[Sampling error: {exc}]"


# 7. MCP Client setup

mcp_client = Client(
    transport=settings.mcp_server_url,
    log_handler=log_handler,
    sampling_handler=sampling_handler,
    auto_initialize=False,
)

# Registries populated after connection
_tools_registry: dict[str, Any] = {}
_resources_registry: dict[str, Any] = {}


async def _connect_and_discover() -> None:
    """Connect to MCP server and discover available tools and
    resources."""
    client_log.info("Connecting to MCP server at %s ...", settings.mcp_server_url)
    async with mcp_client:
        result = await mcp_client.initialize(timeout=settings.initialization_timeout)
        client_log.info("Connected to MCP server: %s", result.serverInfo.name)

        tools_list = await mcp_client.list_tools()
        for t in tools_list:
            _tools_registry[t.name] = t
        tool_names = list(_tools_registry.keys())
        client_log.info(
            "Discovered %d tools: %s",
            len(_tools_registry),
            tool_names,
        )

        resources_list = await mcp_client.list_resource_templates()
        for r in resources_list:
            _resources_registry[r.name] = r
        resource_names = list(_resources_registry.keys())
        client_log.info(
            "Discovered %d resources: %s",
            len(_resources_registry),
            resource_names,
        )
    await _store_entry(
        mcp_interaction_type="tool_invocation",
        component="mcp.client.discovery",
        content=f"Connected to MCP server. Tools: {list(_tools_registry.keys())}",
        metadata={
            "tool_count": len(_tools_registry),
            "resource_count": len(_resources_registry),
        },
    )


# 6. LangChain @tool wrappers around remote MCP capabilities
#    Each wrapper opens a fresh MCP client context, calls the remote
#    tool/resource, and returns the result as a string for the agent.
#       The CRAG pipeline lives inside the
#       knowledge resource itself.


@tool
async def retrieve_domain_context(query: str) -> str:
    """
    Hierarchical CRAG Knowledge Base Tool.

    Queries the MCP server's domain knowledge resource using a full
    Corrective RAG pipeline:
      1. Multi-query expansion of your question
      2. Hierarchical retrieval (domain -> section -> sentence level)
      3. Tree-of-Thought 3-persona relevance evaluation
      4. Tavily web fallback if internal docs are insufficient

    Use this tool FIRST for any factual question. It returns grounded
    context from the knowledge base that you should use to draft your
    answer.

    Args:
        query: Your research question or topic.

    Returns:
        Formatted context string with evaluated knowledge chunks.
    """
    t0 = time.perf_counter()
    client_log.info("Resource read: knowledge://domain/docs/%s", query[:80])

    await _store_entry(
        mcp_interaction_type="resource_read",
        component="mcp.client.resource_read.domain_docs",
        content=f"Resource read request: {query}",
        metadata={"query": query, "session_id": SESSION_ID},
    )

    async def _call() -> str:
        async with mcp_client:
            if not mcp_client.initialize_result:
                await mcp_client.initialize(timeout=settings.initialization_timeout)
            resource_uri = f"knowledge://domain/docs/{query}"
            result = await mcp_client.read_resource(resource_uri)
            return str(result)

    result = await _call()
    latency_ms = int((time.perf_counter() - t0) * 1000)
    client_log.info("Resource returned %d chars of context.", len(result))

    await _store_entry(
        mcp_interaction_type="resource_read",
        component="mcp.client.resource_read.domain_docs.result",
        content=result[:2000],
        metadata={"result_length": len(result), "latency_ms": latency_ms},
    )
    return result[:1500]


@tool
async def reflect_and_correct(
    question: str, draft_answer: str, search_results: str
) -> str:
    """
    Two-Stage Reflection Tool (powered by MCP Sampling).

    Sends your draft answer to the MCP server's Reflection tool.
    The server delegates two LLM calls back to this client via MCP Sampling:

    Stage 1 - Critic: Compares draft against search_results to identify
      hallucinations (claims not in results), contradictions, and omissions.

    Stage 2 - Corrector: Rewrites the draft grounded strictly in
      search_results, fixing every issue the critic identified.

    ALWAYS call this tool before giving a final answer. Pass the raw
    context from retrieve_domain_context as search_results.

    Args:
        question:       The user's original question.
        draft_answer:   Your current draft response.
        search_results: The raw context returned by
            retrieve_domain_context.

    Returns:
        JSON string with keys: critique, corrected_answer, is_sufficient.
    """
    t0 = time.perf_counter()
    client_log.info(
        "Tool call: reflect_and_correct | question='%s'",
        question[:80],
    )

    await _store_entry(
        mcp_interaction_type="tool_invocation",
        component="mcp.client.tool_call.reflect_and_correct",
        content=f"Tool invocation: reflect_and_correct | question: {question[:200]}",
        metadata={
            "tool_name": "reflect_and_correct",
            "question_length": len(question),
            "draft_length": len(draft_answer),
        },
    )

    async def _call() -> str:
        async with mcp_client:
            if not mcp_client.initialize_result:
                await mcp_client.initialize(
                    timeout=settings.initialization_timeout,
                )
            result = await mcp_client.call_tool(
                "reflect_and_correct",
                {
                    "question": question,
                    "draft_answer": draft_answer,
                    "search_results": search_results,
                },
            )
            if isinstance(result, list):
                return "\n".join(
                    (block.text if hasattr(block, "text") else str(block))
                    for block in result
                )
            return str(result)

    result = await _call()
    latency_ms = int((time.perf_counter() - t0) * 1000)
    client_log.info("reflect_and_correct returned %d chars.", len(result))

    await _store_entry(
        mcp_interaction_type="tool_invocation",
        component="mcp.client.tool_call.reflect_and_correct.result",
        content=result[:2000],
        metadata={
            "result_length": len(result),
            "latency_ms": latency_ms,
            "tool_name": "reflect_and_correct",
        },
    )
    return result


# 9. LangChain Agent

AGENT_TOOLS = [retrieve_domain_context, reflect_and_correct]

AGENT_SYSTEM_PROMPT = """
You are a Thinking Agent connected to an enterprise MCP server.

Available tools:
- retrieve_domain_context: Reads the MCP CRAG resource.
- reflect_and_correct: Two-stage Reflection via MCP Sampling
  (Critic + Corrector).

MANDATORY PROTOCOL

PATH A - COMPLEX ANALYSIS / TRADE-OFF QUERIES (evaluate, compare, recommend):

  Step 1: Call retrieve_domain_context with the user's question.
          Store the returned context as your search_results.

  Step 2: Draft a STRUCTURED analysis that:
          - Identifies multiple perspectives or options
          - Lists trade-offs explicitly
          - Cites specific evidence for each claim

  Step 3: Call reflect_and_correct with:
            question       = original user question
            draft_answer   = your Step 2 structured draft
            search_results = exact context from Step 1
          Use the 'corrected_answer' from the JSON result as your final answer.

PATH B - FACTUAL / INFORMATION RETRIEVAL QUERIES (what is, latest, recent):

  Step 1: Call retrieve_domain_context with the user's question.

  Step 2: Draft a CONCISE answer that:
          - Directly answers the question from search results
          - Uses verbatim quotes for key claims when possible
          - Avoids speculation beyond the provided context

  Step 3: Call reflect_and_correct (same args as PATH A).
          Use 'corrected_answer' as your final answer.

ABSOLUTE RULES:
1. ALWAYS call retrieve_domain_context before drafting any answer.
2. ALWAYS call reflect_and_correct before giving a final answer.
3. ALWAYS use 'corrected_answer' from reflect_and_correct as your
   final response.
4. Never present a draft as a final answer without reflection.
5. Ground all factual claims in the context returned by
   retrieve_domain_context.
6. Choose PATH A if the query asks for evaluation/recommendation/comparison.
7. Choose PATH B if the query asks for factual information or current status.
"""


def build_agent():
    """Build and return the LangChain agent with MCP-backed
    tools."""
    client_log.info(
        "Building LangChain agent | tools=%s",
        [t.name for t in AGENT_TOOLS],
    )
    agent = create_agent(
        model=llm,
        tools=AGENT_TOOLS,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    client_log.info("LangChain agent ready.")
    return agent


# 10. Helper: run agent


async def run_agent_query(agent, query: str, label: str = "") -> str:
    """
    Run the agent on a single query and log all steps.

    Args:
        agent: The LangChain agent returned by build_agent().
        query: The user's question.
        label: Optional test label for log headers.

    Returns:
        The agent's final answer as a string.
    """
    sep = "=" * 70
    tag = f" — {label}" if label else ""

    client_log.info("%s", sep)
    client_log.info("AGENT QUERY%s", tag)
    client_log.info("USER: %s", query)
    client_log.info("%s", sep)

    from langchain_core.messages import AIMessage, HumanMessage

    # Log the user query
    await _store_entry(
        mcp_interaction_type="agent_action",
        component="agent.planning.query_received",
        content=f"User query: {query}",
        metadata={"label": label, "query_length": len(query)},
    )

    final_response = ""
    async for event in agent.astream({"messages": [HumanMessage(content=query)]}):
        # Tool call observations
        if "tools" in event:
            for msg in event["tools"].get("messages", []):
                tool_name = getattr(msg, "name", "tool")
                client_log.info(
                    "[OBSERVATION — %s]\n%s",
                    tool_name,
                    str(msg.content),
                )
                await _store_entry(
                    mcp_interaction_type="tool_observation",
                    component=f"agent.tools.observation.{tool_name}",
                    content=str(msg.content)[:2000],
                    metadata={"tool_name": tool_name},
                )

        # Agent reasoning and final answer (after reflection)
        if "model" in event:
            for msg in event["model"].get("messages", []):
                if isinstance(msg, AIMessage) and msg.content.strip():
                    client_log.info("[THOUGHT/FINAL ANSWER]\n%s", msg.content)
                    await _store_entry(
                        mcp_interaction_type="agent_action",
                        component="agent.planning.reflexive_loop",
                        content=str(msg.content)[:2000],
                        metadata={"step": "model_output"},
                    )

    client_log.info("%s", sep)
    return final_response


# 11. Main entry point


async def _async_main() -> None:
    """Async entry point: discover server capabilities then run
    test queries."""

    # Discover tools/resources (optional - wrappers connect on demand)
    await _connect_and_discover()

    # Build the LangChain agent
    agent = build_agent()

    # Test queries
    queries = [
        (
            "What are the current debates around open-source vs "
            "closed-source AI? Evaluate the trade-offs and recommend which "
            "approach is better for long-term AI safety.",
            "Test 1 — Trade-off Query",
        ),
        (
            "What are the latest developments in quantum computing? "
            "Focus on recent breakthroughs and their practical implications.",
            "Test 2 — Factual Query",
        ),
        (
            "Should governments regulate large language models? "
            "Analyse the trade-offs between innovation and public safety.",
            "Test 3 — Policy Query",
        ),
    ]

    for query, label in queries:
        await run_agent_query(agent, query, label)
        # Brief pause between queries
        await asyncio.sleep(2)

    client_log.info("All queries complete. Logs written to mcp_agent_system.log")


def main() -> None:
    """Synchronous entry point called by uv run."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
