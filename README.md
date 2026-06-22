# Thinking Agent — Stage 3: MCP Server / Client / Observability Dashboard

A three-process AI agent system:

- **MCP Server** (`mcp_server`) — exposes a Reflection tool (via MCP Sampling) and a Hierarchical CRAG knowledge resource. Holds no LLM.
- **Agent Client** (`agent_client`) — a LangChain agent acting as the MCP client. Executes all LLM calls (its own reasoning, plus the server's delegated sampling requests) and persists every interaction to a vector-embedded SQLite log store.
- **Analysis Dashboard** (`analysis_dashboard`) — a decoupled Streamlit control plane with its own Log Analysis Agent. Reads the same log store, projects sessions into a Neo4j knowledge graph, and renders latency/token/error charts.

These are three **independent processes** that communicate over the network (MCP) and through shared on-disk state (the SQLite log store) — not three calls in one script. They must be started in **separate terminals**.

---

## Architecture

```
┌────────────────────────────┐        ┌──────────────────────────────────┐
│   AGENT CLIENT (LangChain)  │        │      MCP SERVER (FastMCP)         │
│                              │        │                                    │
│ @tool retrieve_domain_context├───────►│ @resource knowledge://domain/docs │
│ @tool reflect_and_correct   ├───────►│ @tool reflect_and_correct         │
│                              │ http   │   └── ctx.sample() × 2            │
│ sampling_handler() ◄─────────┼────────┤        (NO local LLM on server)  │
│   runs the LLM on the        │ MCP    │   Stage 1: Critic                │
│   server's behalf            │Sampling│   Stage 2: Corrector             │
│                              │        │                                    │
│ log_handler() ─┐             │        └──────────────────────────────────┘
└────────────────┼────────────┘
                  │ every significant event
                  ▼
┌──────────────────────────────────────────┐
│   mcp_agent_log.db  (AsyncSqliteStore)     │   ◄── shared, on-disk, vector-indexed
│   namespace: ("logs", "mcp", ...)          │       log store (HuggingFace embeddings)
└──────────────────────┬─────────────────────┘
                        │ read-only
                        ▼
┌──────────────────────────────────────────┐
│       ANALYSIS DASHBOARD (Streamlit)       │
│                                              │
│  Log Analysis Agent (separate LangChain     │
│  agent, separate process, separate LLM      │
│  conversation from the agent_client)        │
│   ├── search_logs_semantic / list_recent_logs│
│   ├── get_session_trace                      │
│   ├── calculate_latency_trends / token_trends│
│   │   / error_frequency                      │
│   ├── generate_performance_chart (matplotlib)│
│   └── project_session_to_graph → Neo4j       │
└──────────────────────────────────────────┘
```

The dashboard never talks to the MCP server or the agent client directly. It only ever reads `mcp_agent_log.db` and writes to Neo4j. This is what "decoupled" means here: you can run, restart, or crash the dashboard without affecting an in-progress agent session, and vice versa.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.11 | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (Mac/Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` (Windows) |
| Neo4j Aura (or local Neo4j) | any recent | [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura) — free tier is sufficient |

---

## Project Structure

```
thinking_agent_with_mcp/
├── pyproject.toml                 ← uv workspace root (members: mcp_server, agent_client, analysis_dashboard)
├── .env.example                   ← copy to .env
├── .env
├── mcp_agent_system.log           ← generated at runtime — [CLIENT]/[SERVER] flat log
├── mcp_agent_log.db               ← generated at runtime — vector log store (AsyncSqliteStore)
├── analysis_agent.log             ← generated at runtime — dashboard + analysis agent log
├── charts/                        ← generated at runtime — saved PNG charts
├── README.md
├── REFLECTION_STAGE3.md
├── mcp_server/
│   ├── pyproject.toml
│   └── src/mcp_server/
│       └── server.py              ← FastMCP server: CRAG resource + reflect_and_correct tool
├── agent_client/
│   ├── pyproject.toml
│   └── src/agent_client/
│       ├── client.py              ← MCP client + LangChain agent + sampling_handler
│       ├── logging_config.py      ← dual-stream [CLIENT]/[SERVER] flat logger
│       └── log_store.py           ← AsyncSqliteStore wrapper (write_log, search_logs)
└── analysis_dashboard/
    ├── pyproject.toml
    └── src/analysis_dashboard/
        ├── app.py                 ← Streamlit UI (chat, log browser, graph explorer)
        ├── analysis_agent.py      ← decoupled Log Analysis Agent (separate LangChain agent)
        ├── retrieval_tools.py     ← search_logs_semantic, list_recent_logs, get_session_trace
        ├── analytics_tools.py     ← latency/token/error trend tools + chart generation
        ├── neo4j_tools.py         ← project_session_to_graph, query_knowledge_graph, get_graph_summary
        └── settings.py
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
TAVILY_API_KEY=tvly-your-key-here        # optional but recommended
MODEL_NAME=your-model-name-here               # optional, defaults to mistralai/mistral-7b-instruct:free
EMBEDDING_MODEL_NAME=your-embedding-model-name-here  # optional, defaults to sentence-transformers/all-MiniLM-L6-v2
NEO4J_PASSWORD=your-neo4j-password-here
NEO4J_URI=your-neo4j-uri-here
NEO4J_USERNAME=your-neo4j-username-here
LOG_DB_PATH=your-log-db-path-here  # optional, defaults to mcp_agent_log.db
GROQ_API_KEY=your-key-here
```

Get your keys:
- **OpenRouter** (required): https://openrouter.ai/keys
- **Tavily** (optional, free tier): https://app.tavily.com
- **Neo4j Aura** (optional, free tier — only needed for the Graph Explorer tab): https://neo4j.com/cloud/aura

> `.env` lives once, in the **repo root**. All three packages read it from there via `pydantic-settings`' `env_file` config — you do not need a separate `.env` per package.

### 3. Install all dependencies
```bash
uv sync
```
This installs all three workspace members (`mcp-server`, `agent-client`, `analysis-dashboard`) and their dependencies in one shot. No `pip install` needed.

---

## Running — Three Independent Terminals

This stage requires **three concurrent processes**: the server, the agent client, and the observability dashboard. Start them **in this order**, each in its own terminal, from the **repo root**.

### Terminal 1 — MCP Server

<table>
<tr><th>bash / sh (macOS, Linux, WSL)</th></tr>
<tr><td>

```bash
uv run --package mcp-server start-server
```

</td></tr>
<tr><th>cmd.exe (Windows)</th></tr>
<tr><td>

```cmd
uv run --package mcp-server start-server
```

</td></tr>
<tr><th>PowerShell (Windows)</th></tr>
<tr><td>

```powershell
uv run --package mcp-server start-server
```

</td></tr>
</table>

Wait for:
```
[SERVER] [INFO] Starting ThinkingAgent MCP Server on http://localhost:8080/mcp
```

Verify it's up:
```bash
curl http://localhost:8080/health
# Expected output: OK
```

### Terminal 2 — Agent Client

<table>
<tr><th>bash / sh</th></tr>
<tr><td>

```bash
uv run --package agent-client start-agent
```

</td></tr>
<tr><th>cmd.exe</th></tr>
<tr><td>

```cmd
uv run --package agent-client start-agent
```

</td></tr>
<tr><th>PowerShell</th></tr>
<tr><td>

```powershell
uv run --package agent-client start-agent
```

</td></tr>
</table>

This connects to the server, runs the built-in test queries, and writes:
- `mcp_agent_system.log` — flat, human-readable `[CLIENT]`/`[SERVER]` stream
- `mcp_agent_log.db` — vector-embedded, hierarchically-namespaced log store

Let it finish (`All queries complete...`) at least once before moving to Terminal 3, so the dashboard has data to analyse.

### Terminal 3 — Observability / Analysis Dashboard

The dashboard is a Streamlit app, so it must be launched through the `streamlit` CLI rather than its `start-dashboard` script entrypoint — `uv run --package ... <script>` invokes the entrypoint as a plain Python function, which does not give Streamlit the script-run context it needs to serve a UI.

<table>
<tr><th>bash / sh</th></tr>
<tr><td>

```bash
uv run --package analysis-dashboard streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```

</td></tr>
<tr><th>cmd.exe</th></tr>
<tr><td>

```cmd
uv run --package analysis-dashboard streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```

</td></tr>
<tr><th>PowerShell</th></tr>
<tr><td>

```powershell
uv run --package analysis-dashboard streamlit run analysis_dashboard/src/analysis_dashboard/app.py
```

</td></tr>
</table>

This opens `http://localhost:8501` in your browser. Use the sidebar quick actions or the chat box to query the Log Analysis Agent (e.g. *"Analyse recent logs and chart latency trends"*, *"Sync the latest session to Neo4j"*).

> Run order matters only loosely: the dashboard will start fine before Terminal 1/2, but its quick actions and chat queries won't have anything to analyse until `mcp_agent_log.db` has at least one session in it.

---

## Watching Logs

Two log surfaces exist side by side, generated by different processes:

```bash
# Terminal 1/2 activity — agent_client + forwarded MCP server notifications
tail -f mcp_agent_system.log

# Terminal 3 activity — Streamlit dashboard + Log Analysis Agent
tail -f analysis_agent.log
```

Example `mcp_agent_system.log` excerpt:
```
[2026-06-19 09:00:05] [CLIENT] [INFO] Tool call: retrieve_domain_context
[2026-06-19 09:00:05] [SERVER] [INFO] CRAG resource invoked for query: '...'
[2026-06-19 09:00:06] [SERVER] [INFO] Hierarchical retrieval L1 complete — 2 relevant domain(s)
[2026-06-19 09:00:08] [CLIENT] [INFO] MCP Sampling request received from server
[2026-06-19 09:00:15] [SERVER] [INFO] Reflection complete | is_sufficient=True
```

The structured equivalent of every line above also lands in `mcp_agent_log.db`, queryable by the dashboard's `search_logs_semantic` and `list_recent_logs` tools.

---

## How It Works

### CRAG Resource (`knowledge://domain/docs/{query}`)
When the agent calls `retrieve_domain_context`, the server runs:
1. **Multi-query expansion** — rule-based variants grounded in the original query's own terms
2. **Hierarchical retrieval** — domain → section → sentence tree, each level backed by an inverted index
3. **ToT 3-perspective evaluation** — relevance, accuracy, coverage scoring per chunk, via MCP Sampling
4. **Tavily fallback** — web search if internal docs are insufficient

### Reflection Tool (`reflect_and_correct`)
The server holds **no LLM**. Both stages delegate to the client via `ctx.sample()`:
1. **Critic** — compares draft against search results, flags hallucinations/contradictions/omissions
2. **Corrector** — rewrites the draft using search results as ground truth

### Vector Log Store (`agent_client/log_store.py`)
Every significant event — tool calls, resource reads, sampling requests/responses, agent reasoning steps — is validated against a `LogEntry` pydantic schema and written to `mcp_agent_log.db` via `langgraph.store.sqlite.aio.AsyncSqliteStore`, indexed under a hierarchical dot-separated namespace (e.g. `("logs", "mcp", "sampling", "request")`) with HuggingFace sentence-transformer embeddings on the `content` field.

### Log Analysis Agent (`analysis_dashboard/analysis_agent.py`)
A second, independent LangChain agent — its own process, its own conversation history, no shared state with `agent_client` other than the log database it reads. Given a natural-language diagnostic question, it chains retrieval tools (`search_logs_semantic`, `list_recent_logs`, `get_session_trace`) into analytics tools (`calculate_latency_trends`, `calculate_token_trends`, `calculate_error_frequency`, `generate_performance_chart`) and/or Neo4j tools (`project_session_to_graph`, `query_knowledge_graph`, `get_graph_summary`).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on agent client start | Start the MCP server first (Terminal 1) and wait for the ready log line |
| `ModuleNotFoundError` | Run `uv sync` from the repo root |
| `OPENROUTER_API_KEY not found` | Ensure `.env` is in the repo root (next to the root `pyproject.toml`) |
| Tavily fallback skipped | Set `TAVILY_API_KEY` in `.env` |
| Sampling requests failing | Ensure `sampling_handler` is passed to `Client()` before `initialize()` (already wired in `client.py`) |
| Dashboard says "No latency/token/error data" | Run the agent client (Terminal 2) at least once first — the dashboard only reads, it never generates traffic |
| Streamlit shows a blank page / `missing ScriptRunContext` warnings | You launched via `start-dashboard` instead of `streamlit run` — use the Terminal 3 command above |
| Graph Explorer tab shows "Neo4j not connected" | Set `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` in `.env`; this is optional and the rest of the dashboard works without it |
| `calculate_token_trends` returns `no_data` for old sessions | Token metadata (`token_count`, `prompt_tokens`, `completion_tokens`) is only captured in sampling responses logged **after** the corresponding fix in `client.py`'s `sampling_handler` — re-run the agent client to generate fresh sessions with token metadata |