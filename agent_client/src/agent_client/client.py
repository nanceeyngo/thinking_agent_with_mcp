"""
agent_client/client.py

MCP Client + LangChain AI Agent.

Responsibilities:
  1. Connect to the FastMCP server over streamable-http
  2. Wrap remote MCP tools as local LangChain @tool functions
  3. Handle incoming MCP Sampling requests (execute LLM calls on behalf
     of server)
  4. Initialize LangChain create_agent with wrapped tools
  5. Write dual-stream logs ([CLIENT] / [SERVER]) to agent_system.log
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import Client
from fastmcp.client.logging import LogMessage
from fastmcp.client.sampling import SamplingMessage, SamplingParams
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from mcp.shared.context import RequestContext
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_client.logging_config import (
    get_server_log_writer,
    setup_client_logger,
)

# 1. Settings


class Settings(BaseSettings):
    openrouter_api_key: SecretStr
    tavily_api_key: SecretStr | None = None
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


# 3. LLM (used locally for sampling AND for the agent)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model=settings.model_name,
    temperature=settings.model_temperature,
    openai_api_key=settings.openrouter_api_key.get_secret_value(),
    openai_api_base=OPENROUTER_BASE_URL,
    max_tokens=2048,
    default_headers={
        "HTTP-Referer": "https://thinking-agent-stage2",
        "X-Title": "Thinking Agent Stage 2",
    },
)

client_log.info(
    "LLM initialised: model=%s via OpenRouter",
    settings.model_name,
)

# 4. MCP Callback Handlers


LOGGING_LEVEL_MAP = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}


async def log_handler(message: LogMessage) -> None:
    """
    Receive MCP log notifications forwarded from the server.
    Write them to agent_system.log with [SERVER] prefix so they are
    clearly distinct from [CLIENT] entries.
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


async def sampling_handler(
    messages: list[SamplingMessage],
    params: SamplingParams,
    context: RequestContext,
) -> str:
    """
    Handle MCP Sampling requests routed from the server's Reflection tool.

    The server calls ctx.request_sampling() which triggers this handler.
    We execute the LLM call here using our locally configured ChatOpenAI
    instance and return the generated text back to the server over the
    transport layer.

    This is the 'Sampling Paradox': the server requests an LLM
    completion from the client rather than hosting its own model.

    Args:
        messages: List of SamplingMessage objects, where each message has
                 'role' (string: 'user'|'assistant'|'system') and
                 'content' (TextContent with 'type' and 'text' fields).
        params: SamplingParams with temperature, max_tokens, etc.
        context: MCP RequestContext for logging.

    Returns:
        String response from the LLM.
    """
    client_log.info(
        "MCP Sampling request received from server | request_id=%s",
        context.request_id,
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

    # If no system message in messages, but we have text to process,
    # treat the first user message as the main prompt
    # (system message if provided should take precedence)

    client_log.debug(
        "Sampling request converted | messages=%d | system_msg=%s | "
        "max_tokens=%s | temperature=%s",
        len(messages),
        "present" if system_message else "absent",
        getattr(params, "maxTokens", "default"),
        getattr(params, "temperature", "default"),
    )

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
        client_log.info(
            "Sampling response generated | length=%d chars", len(result_text)
        )
        return result_text
    except Exception as exc:
        client_log.error("Sampling LLM call failed: %s", exc)
        return f"[Sampling error: {exc}]"


# 5. MCP Client setup

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
      2. Hierarchical retrieval (domain → section → sentence level)
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
    client_log.info("Resource read: knowledge://domain/docs/%s", query[:80])

    async def _call() -> str:
        async with mcp_client:
            if not mcp_client.initialize_result:
                await mcp_client.initialize(timeout=settings.initialization_timeout)
            resource_uri = f"knowledge://domain/docs/{query}"
            result = await mcp_client.read_resource(resource_uri)
            return str(result)

    result = await _call()
    client_log.info("Resource returned %d chars of context.", len(result))
    return result


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
    client_log.info(
        "Tool call: reflect_and_correct | question='%s'",
        question[:80],
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
    client_log.info("reflect_and_correct returned %d chars.", len(result))
    return result


# 7. LangChain Agent

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


# 8. Helper: run agent on a query and print full logs


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

        # Agent reasoning and final answer (after reflection)
        if "model" in event:
            for msg in event["model"].get("messages", []):
                if isinstance(msg, AIMessage) and msg.content.strip():
                    client_log.info("[THOUGHT/FINAL ANSWER]\n%s", msg.content)

    client_log.info("%s", sep)
    return final_response


# 9. Main entry point


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

    client_log.info("All queries complete. Logs written to agent_system.log")


def main() -> None:
    """Synchronous entry point called by uv run."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
