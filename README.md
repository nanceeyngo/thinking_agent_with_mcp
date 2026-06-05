# Thinking Agent — Stage 2: MCP Server/Client

A modular AI agent system where a **FastMCP Server** exposes a Reflection tool (via MCP Sampling) and a Hierarchical CRAG knowledge resource, and a **LangChain Agent** acts as the MCP Client.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   AGENT CLIENT                        │
│  LangChain create_agent                               │
│    ├── @tool: retrieve_domain_context  ─────────────┐ │
│    └── @tool: reflect_and_correct      ─────────────┤ │
│  sampling_handler (runs LLM for server) ◄────────────┘ │
│  log_handler → writes [SERVER] logs to agent_system.log│
└──────────────────────┬───────────────────────────────┘
                       │ streamable-http
                       │ http://localhost:8080/mcp
┌──────────────────────▼───────────────────────────────┐
│                   MCP SERVER (FastMCP)                 │
│  @resource knowledge://domain/docs/{query}             │
│    └── CRAG: multi-query → hierarchical retrieval      │
│              → ToT 3-persona eval → Tavily fallback    │
│  @tool reflect_and_correct                             │
│    └── ctx.sample() × 2  (NO local LLM on server)     │
│         Stage 1: Critic  → delegates to client LLM    │
│         Stage 2: Corrector → delegates to client LLM  │
└──────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.11 | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (Mac/Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` (Windows) |

---

## Project Structure

```
thinking_agent_with_mcp/
├── pyproject.toml                        ← uv workspace root
├── .env.example                          ← copy to .env
├── agent_system.log                      ← generated at runtime
├── README.md
├── REFLECTION.md
├── mcp_server/
│   ├── pyproject.toml
│   └── src/mcp_server/
│       └── server.py                     ← FastMCP server
└── agent_client/
    ├── pyproject.toml
    └── src/agent_client/
        ├── client.py                     ← MCP client + LangChain agent
        └── logging_config.py             ← dual-stream logger
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/nanceeyngo/thinking_agent_with_mcp.git
cd thinking_agent_with_mcp
```

### 2. Create your `.env` file
```bash
cp .env.example .env
```

Open `.env` and fill in your keys:
```env
OPENROUTER_API_KEY=sk-or-your-key-here
TAVILY_API_KEY=tvly-your-key-here          # optional — enables web fallback
MODEL_NAME=nvidia/nemotron-3-super-120b-a12b:free  # free model, no credit needed
```

Get your keys:
- **OpenRouter** (required): https://openrouter.ai/keys
- **Tavily** (optional, free tier): https://app.tavily.com

### 3. Install all dependencies
```bash
uv sync
```

This installs both packages and all dependencies. No `pip install` needed.

---

## Running

Open **two separate terminal windows** in the project root.

### Terminal 1 — Start the MCP Server
```bash
uv run --package mcp-server start-server
```

Wait until you see:
```
[SERVER] [INFO] Starting ThinkingAgent MCP Server on http://localhost:8080/mcp
```

Verify it's running:
```bash
curl http://localhost:8080/health
# Expected output: OK
```

### Terminal 2 — Start the Agent Client
```bash
uv run --package agent-client start-agent
```

The agent will:
1. Connect to the server and discover tools/resources
2. Run test queries demonstrating the full pipeline
3. Write all logs to `agent_system.log`

---

## Watching Logs

All output is written to `agent_system.log` with clear prefixes:

```
[2026-05-31 09:00:05] [CLIENT] [INFO] Tool call: retrieve_domain_context
[2026-05-31 09:00:05] [SERVER] [INFO] CRAG pipeline initiated for query: '...'
[2026-05-31 09:00:05] [SERVER] [DEBUG] ToT evaluation complete — kept 4/8 chunks
[2026-05-31 09:00:08] [CLIENT] [INFO] MCP Sampling request received from server
[2026-05-31 09:00:15] [SERVER] [INFO] Reflection complete | is_sufficient=True
```

```bash
# Watch live
tail -f agent_system.log
```

---

## How It Works

### CRAG Resource (`knowledge://domain/docs/{query}`)
When the agent calls `retrieve_domain_context`, the server runs:
1. **Multi-query expansion** — generates semantic query variants
2. **Hierarchical retrieval** — searches domain → section → sentence levels
3. **ToT 3-persona evaluation** — Analytical + Relevance + Coverage scoring per chunk
4. **Tavily fallback** — web search if internal docs score below threshold

### Reflection Tool (`reflect_and_correct`)
The server holds **no LLM**. Both stages delegate to the client via `ctx.sample()`:
1. **Critic** — compares draft against search results, flags hallucinations
2. **Corrector** — rewrites draft using search results as ground truth

The client's `sampling_handler` intercepts these requests and executes the LLM locally.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on client start | Start the server first and wait for the ready log line |
| `ModuleNotFoundError` | Run `uv sync` from the repo root |
| `OPENROUTER_API_KEY not found` | Ensure `.env` is in the repo root (next to `pyproject.toml`) |
| Tavily fallback skipped | Set `TAVILY_API_KEY` in `.env` |
| Sampling requests failing | Ensure `sampling_handler` is passed to `Client()` constructor before `initialize()` |