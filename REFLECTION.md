# Reflection

## The Architecture Shift

Moving from local LangChain tools to a decoupled MCP-based architecture introduced clear operational benefits. The MCP server became responsible for domain knowledge retrieval, hierarchical CRAG processing, and reflection workflows, while the client focused on agent orchestration and LLM execution. This separation improved modularity, reusability, and maintainability because retrieval logic could evolve independently of the agent implementation. It also created a cleaner boundary between knowledge services and reasoning services, making the system easier to extend with additional tools and resources.

However, the architecture introduced performance bottlenecks. Every tool invocation now requires network communication between client and server, increasing latency compared to local function calls. The reflection workflow was particularly expensive because it involved two separate sampling requests (Critic and Corrector), each requiring a round trip between the server and client. During testing, retrieval completed quickly, but reflection added noticeable delays due to sequential LLM calls and transport overhead.

## The Sampling Paradox

Implementing MCP Sampling was one of the most interesting aspects of the project. Instead of the server hosting an LLM, the server requested generations from the client through `ctx.sample()`. This reversed the traditional architecture and allowed the server to remain model-agnostic and free of API keys. The same MCP server could therefore work with different client-side models without modification.

The server's reflection tool now uses a structured critic/corrector workflow. The critic returns JSON including `problems`, `evidence`, `reason`, and `is_sufficient`, allowing the server to decide whether correction is actually needed before issuing another sampling request.

From a security perspective, this reduces the need to expose model credentials on the server. However, it also creates trust considerations because the server depends on the client to execute sampling requests faithfully. Structurally, it reinforces MCP’s philosophy that servers provide capabilities while clients provide intelligence.

## State & Context Management

Implementing hierarchical chunking, multi-query expansion, and Tree-of-Thought evaluation inside an MCP resource significantly simplified the agent. Rather than managing retrieval pipelines inside LangChain, the agent only needed to request domain context through a single resource call. The resource returned already-expanded, filtered, and evaluated knowledge.

The current ToT evaluation is implemented via LLM-guided scoring rather than a purely heuristic rule-based filter. This means the server now delegates relevance assessment to the client-side model when evaluating candidate chunks.

This shifted context management away from the agent and into the retrieval layer. The agent operated on curated context instead of raw documents, reducing prompt complexity and making reflection more effective. Testing also revealed the importance of retrieval precision: an irrelevant AI Safety chunk was retrieved during a Quantum Computing query, demonstrating that retrieval quality directly influences downstream reflection and answer grounding.
