"""
mcp_server/server.py

FastMCP Server exposing:
  1. @tool  - reflect_and_correct   : Two-stage Reflection via MCP Sampling
                                       (server holds NO LLM - delegates to client)
  2. @resource - knowledge://domain/docs : Hierarchical CRAG with ToT evaluation
                                           and Tavily fallback

Transport: streamable-http  (http://localhost:8080/mcp)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from mcp.types import SamplingMessage, TextContent

load_dotenv()

# 1. Logging setup
#    Logs go to server stdout with timestamps.
#    ctx.info() / ctx.debug() forward entries to the client via MCP notifications.

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [SERVER] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 2. FastMCP initialisation

mcp = FastMCP(
    name="ThinkingAgentServer",
    instructions=(
        "This server exposes a Reflection tool (uses MCP Sampling) "
        "and a Hierarchical CRAG knowledge resource."
    ),
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(_: Request) -> PlainTextResponse:
    """Lightweight health-check endpoint for load balancers / readiness probes."""
    return PlainTextResponse("OK")


# 3. Domain Knowledge Base (used by the CRAG resource)
#    Hierarchical structure: sections -> paragraphs -> sentences

DOMAIN_KNOWLEDGE: dict[str, dict[str, list[str]]] = {
    "AI Safety": {
        "Open vs Closed Source": [
            "Open-source AI models promote transparency and allow the research "
            "community to audit weights, training data, and inference code.",
            "Closed-source AI systems give organisations tighter control over "
            "misuse but reduce external accountability and peer review.",
            "A hybrid approach - open weights with restricted fine-tuning APIs - "
            "is emerging as a practical middle ground for safety-conscious deployment.",
            "The EU AI Act treats open-source foundation models differently from "
            "closed proprietary ones, applying lighter obligations to the former.",
        ],
        "Alignment Techniques": [
            "Reinforcement Learning from Human Feedback (RLHF) remains the dominant "
            "alignment technique but is sensitive to reward hacking.",
            "Constitutional AI (CAI) uses a set of principles to guide self-critique "
            "and revision, reducing reliance on human labellers.",
            "Scalable oversight research investigates how humans can supervise AI "
            "systems that are smarter than themselves.",
        ],
    },
    "Quantum Computing": {
        "Recent Breakthroughs": [
            (
                "Google's Willow chip (2024) demonstrated below-threshold error "
                "correction, a milestone toward fault-tolerant quantum computing."
            ),
            "IBM's 1000+ qubit Condor processor represents the largest superconducting "
            "quantum processor publicly demonstrated.",
            "Microsoft announced topological qubits in 2025, claiming inherently lower "
            "error rates than transmon architectures.",
        ],
        "Practical Implications": [
            "Quantum advantage for optimisation problems relevant to logistics and "
            "drug discovery is expected within 5-10 years.",
            "Post-quantum cryptography standards (NIST PQC) are being finalised to "
            "protect classical infrastructure from future quantum attacks.",
        ],
    },
    "LLM Regulation": {
        "Government Approaches": [
            "The EU AI Act classifies general-purpose AI models above 10^25 FLOPs "
            "as high-capability and subjects them to systemic risk obligations.",
            "The US Executive Order on AI (Oct 2023) requires dual-use foundation "
            "model developers to share safety test results with the government.",
            "China's Interim Measures for Generative AI require algorithm registration "
            "and content moderation for publicly deployed LLMs.",
        ],
        "Industry Self-Regulation": [
            "Frontier AI companies including Anthropic, Google DeepMind, and OpenAI "
            "signed voluntary safety commitments at the UK AI Safety Summit (2023).",
            "Model cards and system cards have become informal standards for "
            "disclosing capabilities, limitations, and intended use cases.",
        ],
    },
}


# 4. CriticReport model for structured output from the Critic LLM in the reflection loop
class CriticReport(BaseModel):
    """
    Structured output returned by the Critic LLM.
    """

    problems: list[str] = Field(default_factory=list)

    is_sufficient: bool = False

    reason: str = ""

    evidence: list[str] = Field(default_factory=list)


# 5. Internal helpers for CRAG


# helper to build a hierarchical index from the nested DOMAIN_KNOWLEDGE structure.
def _build_hierarchical_index() -> list[dict[str, Any]]:
    """
    Flatten the nested DOMAIN_KNOWLEDGE into a list of chunks,
    each carrying its full hierarchical path as metadata.

    Hierarchy: domain -> section -> sentence (index)
    This mirrors a real enterprise doc system: category -> chapter -> paragraph.
    """
    chunks: list[dict[str, Any]] = []
    for domain, sections in DOMAIN_KNOWLEDGE.items():
        # Level 1: domain summary (high-level chunk)
        domain_summary = f"Domain: {domain}. Sections: {', '.join(sections.keys())}."
        chunks.append(
            {
                "level": "domain",
                "domain": domain,
                "section": None,
                "text": domain_summary,
                "id": f"{domain}::summary",
            }
        )
        for section, sentences in sections.items():
            # Level 2: section summary
            section_summary = (
                f"Section '{section}' in domain '{domain}': "
                + " | ".join(sentences[:2])
            )
            chunks.append(
                {
                    "level": "section",
                    "domain": domain,
                    "section": section,
                    "text": section_summary,
                    "id": f"{domain}::{section}::summary",
                }
            )
            # Level 3: individual sentences (granular)
            for idx, sentence in enumerate(sentences):
                chunks.append(
                    {
                        "level": "sentence",
                        "domain": domain,
                        "section": section,
                        "text": sentence,
                        "id": f"{domain}::{section}::{idx}",
                    }
                )
    return chunks


HIERARCHICAL_INDEX: list[dict[str, Any]] = _build_hierarchical_index()
logger.info(
    "Hierarchical index built — %d chunks across %d domains.",
    len(HIERARCHICAL_INDEX),
    len(DOMAIN_KNOWLEDGE),
)


# helper for CRAG resource - multi-query expansion to generate
# semantic variants of the raw query
def _expand_query(raw_query: str) -> list[str]:
    """
    Multi-query expansion: generate semantic variants of the raw query.
    In production this would call an LLM; here we use rule-based expansion
    to avoid a circular dependency (server must not hold its own LLM).
    """
    q = raw_query.lower().strip()
    variants = [raw_query]

    # Simple keyword-based expansion
    if any(k in q for k in ["open", "closed", "source"]):
        variants.append("open-source vs closed-source AI trade-offs")
        variants.append("AI transparency and safety model comparison")
    if any(k in q for k in ["quantum", "qubit", "computing"]):
        variants.append("quantum computing breakthroughs 2024 2025")
        variants.append("fault-tolerant quantum error correction")
    if any(k in q for k in ["regulat", "govern", "law", "policy"]):
        variants.append("AI regulation government policy LLM")
        variants.append("EU AI Act large language model compliance")
    if any(k in q for k in ["align", "safety", "rlhf", "constitutional"]):
        variants.append("AI alignment techniques RLHF constitutional AI")
        variants.append("scalable oversight AI safety research")

    # Always add the raw query keywords as a fallback variant
    keywords = [w for w in q.split() if len(w) > 4]
    if keywords:
        variants.append(" ".join(keywords[:4]))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


# helper for CRAG resource - performs hierarchical
# retrieval based on the raw query
def _hierarchical_retrieve(raw_query: str, ctx: Context) -> list[dict[str, Any]]:
    """
    True hierarchical retrieval (3-level indexing):

    Level 1: Search across all domains to identify top_k relevant domains
    Level 2: For each relevant domain, search sections to identify top_k
    Level 3: For each relevant section, search sentences to identify top_k

    This avoids searching the entire store on every retrieval by
    progressively narrowing the search space at each level.

    Returns: List of top sentence-level chunks with full hierarchical context.
    """
    q_words = set(raw_query.lower().split())
    level_weights = {"sentence": 1.5, "section": 1.0, "domain": 0.5}
    top_k = 3

    # LEVEL 1: Domain-level search
    logger.debug("Hierarchical retrieval Level 1: Searching domains...")
    domain_scores: dict[str, float] = {}

    for chunk in HIERARCHICAL_INDEX:
        if chunk["level"] != "domain":
            continue
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(q_words & chunk_words)
        if overlap > 0:
            domain_name = chunk["domain"]
            score = overlap * level_weights["domain"]
            domain_scores[domain_name] = domain_scores.get(domain_name, 0) + score

    # Select top_k relevant domains
    top_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]
    selected_domain_names = [d[0] for d in top_domains]
    if not selected_domain_names:
        selected_domain_names = [
            chunk["domain"]
            for chunk in HIERARCHICAL_INDEX
            if chunk["level"] == "domain"
        ]
        logger.debug(
            "Level 1 yielded no strong domain matches; using all domains: %s",
            selected_domain_names,
        )
    else:
        logger.debug(
            "Level 1 complete — found %d relevant domains: %s",
            len(selected_domain_names),
            selected_domain_names,
        )

    # LEVEL 2: Section-level search within top domains
    logger.debug("Hierarchical retrieval Level 2: Searching sections in top domains...")
    section_scores: dict[tuple[str, str], float] = {}

    for chunk in HIERARCHICAL_INDEX:
        if chunk["level"] != "section":
            continue
        if chunk["domain"] not in selected_domain_names:
            continue
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(q_words & chunk_words)
        if overlap > 0:
            key = (chunk["domain"], chunk["section"])
            score = overlap * level_weights["section"]
            section_scores[key] = section_scores.get(key, 0) + score

    # Select top_k relevant sections per domain
    top_sections = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)[
        : top_k * max(1, len(selected_domain_names))
    ]
    selected_sections = {key for key, _ in top_sections}
    if not selected_sections:
        selected_sections = {
            (chunk["domain"], chunk["section"])
            for chunk in HIERARCHICAL_INDEX
            if chunk["level"] == "section" and chunk["domain"] in selected_domain_names
        }
    logger.debug(
        "Level 2 complete — found %d relevant sections",
        len(selected_sections),
    )

    # LEVEL 3: Sentence-level search within top sections
    logger.debug(
        "Hierarchical retrieval Level 3: " "Searching sentences in top sections..."
    )
    final_chunks: list[dict[str, Any]] = []

    for chunk in HIERARCHICAL_INDEX:
        if chunk["level"] != "sentence":
            continue
        key = (chunk["domain"], chunk["section"])
        if key not in selected_sections:
            continue
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(q_words & chunk_words)
        if overlap > 0:
            score = overlap * level_weights["sentence"]
            final_chunks.append((score, chunk))

    # Sort by score and take top_k
    final_chunks.sort(key=lambda x: x[0], reverse=True)
    result = [chunk for _, chunk in final_chunks[: top_k * len(selected_sections)]]
    logger.debug(
        "Level 3 complete — retrieved %d relevant sentences",
        len(result),
    )

    return result


# helper for CRAG resource - Tree-of-Thought evaluation using LLM via MCP Sampling
async def _tot_evaluate_with_llm(
    query: str,
    chunks: list[dict[str, Any]],
    ctx: Context,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Tree-of-Thought evaluation using LLM via MCP Sampling.

    Instead of rule-based scoring, the LLM acts as three personas:
    - Persona A (Analytical): Evaluates technical accuracy and depth
    - Persona B (Relevance): Evaluates how well chunks answer the query
    - Persona C (Coverage): Evaluates whether chunks provide complete information

    The LLM produces structured JSON indicating which chunks to keep and
    whether fallback is needed.

    Returns: (selected_chunks, fallback_needed)
    """
    if not chunks:
        return [], True

    logger.debug("ToT evaluation: Requesting LLM evaluation via sampling...")

    # Format chunks for evaluation
    chunk_list = "\n".join(
        [
            f"{i}. [{chunk['level'].upper()} | {chunk['domain']}"
            + (f" / {chunk['section']}" if chunk["section"] else "")
            + f"]: {chunk['text'][:100]}..."
            for i, chunk in enumerate(chunks)
        ]
    )

    tot_messages = [
        SamplingMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text=(
                    "You are evaluating retrieved chunks as three personas: "
                    "Analyst (depth/accuracy), Relevance Expert (query match), "
                    "and Coverage Reviewer (completeness). Produce a JSON report."
                ),
            ),
        ),
        SamplingMessage(
            role="user",
            content=TextContent(
                type="text",
                text=(
                    "Evaluate these chunks using three perspectives:\n\n"
                    f"QUERY: {query}\n\n"
                    f"CHUNKS:\n{chunk_list}\n\n"
                    "For each chunk, assign scores 0-1 for each persona. "
                    "Output JSON:\n"
                    "{\n"
                    '  "evaluations": [{"id": 0, "analytical_score": 0.8, '
                    '"relevance_score": 0.9, "coverage_score": 0.7}],\n'
                    '  "fallback_needed": false,\n'
                    '  "reasoning": "..."\n'
                    "}"
                ),
            ),
        ),
    ]

    try:
        tot_result = await ctx.sample(
            messages=tot_messages,
            max_tokens=1024,
            temperature=0.0,
        )

        if hasattr(tot_result, "text"):
            tot_text = tot_result.text
        else:
            tot_text = str(tot_result)

        # Parse LLM response
        import json as json_lib

        try:
            tot_data = json_lib.loads(tot_text)
            evaluations = tot_data.get("evaluations", [])
            fallback_needed = tot_data.get("fallback_needed", True)

            # Keep chunks with average score >= 0.5
            kept_chunk_ids = {
                e["id"]
                for e in evaluations
                if (
                    e.get("analytical_score", 0)
                    + e.get("relevance_score", 0)
                    + e.get("coverage_score", 0)
                )
                / 3
                >= 0.5
            }

            selected = [c for i, c in enumerate(chunks) if i in kept_chunk_ids]

            logger.info(
                "ToT evaluation complete — kept %d/%d chunks | fallback=%s",
                len(selected),
                len(chunks),
                fallback_needed,
            )
            await ctx.info(
                f"ToT evaluation complete — kept {len(selected)}/{len(chunks)} chunks"
            )

            return selected, fallback_needed
        except (json_lib.JSONDecodeError, KeyError):
            # Fallback to original chunks if parsing fails
            logger.warning("ToT LLM response not valid JSON, using all chunks")
            return chunks, True

    except Exception as exc:
        logger.error("ToT LLM evaluation failed: %s", exc)
        await ctx.error(f"ToT evaluation failed: {exc}")
        return chunks, True


# helper for CRAG resource - multi-query expansion with
# hierarchical retrieval for each variant
def _multi_query_retrieve_hierarchical(
    raw_query: str, ctx: Context
) -> list[dict[str, Any]]:
    """
    Multi-query expansion with hierarchical retrieval for each variant.
    """
    variants = _expand_query(raw_query)
    logger.debug(
        "Multi-query expansion produced %d variants: %s", len(variants), variants
    )

    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []

    for variant in variants:
        results = _hierarchical_retrieve(variant, ctx)
        for chunk in results:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                merged.append(chunk)

    logger.debug(
        "Multi-query hierarchical retrieval merged %d unique chunks.",
        len(merged),
    )
    return merged


# helper for CRAG resource - Tavily web search
# fallback when internal docs are insufficient
async def _tavily_fallback(query: str, ctx: Context) -> list[str]:
    """Run a Tavily web search as fallback when internal docs are insufficient."""
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key:
        await ctx.warning("TAVILY_API_KEY not set — skipping web fallback.")
        return []

    try:
        # Use langchain-tavily directly (no LLM dependency)
        from langchain_tavily import TavilySearch

        search = TavilySearch(
            max_results=3,
            tavily_api_key=tavily_api_key,
        )
        raw = search.invoke(query)

        # TavilySearch returns a list of result dicts or a string
        if isinstance(raw, list):
            return [
                f"[WEB | {r.get('url', 'unknown')}]\n{r.get('content', '')}"
                for r in raw
                if r.get("content")
            ]
        return [f"[WEB]\n{raw}"]

    except Exception as exc:
        logger.error("Tavily fallback error: %s", exc)
        await ctx.error(f"Tavily fallback failed: {exc}")
        return []


# helper for critic/corrector tool - runs the Critic LLM via
# MCP Sampling to evaluate the draft answer
async def _run_critic(
    question: str,
    draft_answer: str,
    search_results: str,
    ctx: Context,
) -> CriticReport:
    """
    Run the Critic LLM via MCP Sampling to evaluate the draft answer.

    Args:
        question:       The user's original question.
        draft_answer:         The agent's current draft answer.
        search_results: Raw retrieved context (from CRAG resource).
        ctx:            FastMCP context for logging and sampling.

    Returns:
        CriticReport with identified problems, sufficiency flag, and evidence.
    """
    logger.info("Running critic evaluation via MCP Sampling...")
    await ctx.info("Running critic evaluation via MCP Sampling...")
    critic_messages = [
        SamplingMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text=(
                    "You are a rigorous fact-checking critic.\n"
                    "Evaluate the draft answer ONLY against the supplied "
                    "search results.\n"
                    "Return valid JSON only."
                ),
            ),
        ),
        SamplingMessage(
            role="user",
            content=TextContent(
                type="text",
                text=(f"""
                    QUESTION:
                    {question}

                    SEARCH RESULTS:
                    {search_results}

                    DRAFT ANSWER:
                    {draft_answer}

                    Return ONLY valid JSON:

                    {{
                    "problems": [
                        "issue1",
                        "issue2"
                    ],
                    "is_sufficient": true,
                    "reason": "brief explanation",
                    "evidence": [
                        "specific supporting citation"
                    ]
                    }}

                    Rules:

                    - Hallucinations = unsupported claims
                    - Contradictions = claims conflicting with results
                    - Omissions = important missing information
                    - If no issues exist:
                        problems=[]
                        is_sufficient=true
                    """),
            ),
        ),
    ]

    result = await ctx.sample(
        messages=critic_messages,
        max_tokens=1000,
        temperature=0.0,
    )

    if hasattr(result, "text"):
        text = result.text
    else:
        text = str(result)

    logger.info("Critic response:\n%s", text)
    try:
        data = json.loads(text)

        return CriticReport(
            problems=data.get("problems", []),
            is_sufficient=data.get("is_sufficient", False),
            reason=data.get("reason", ""),
            evidence=data.get("evidence", []),
        )
    except Exception:
        logger.exception("Critic returned invalid JSON")
        return CriticReport(
            problems=[text],
            is_sufficient=False,
            reason="Critic returned invalid JSON",
            evidence=[],
        )


# helper for critic/corrector tool - runs the Corrector LLM via MCP Sampling
# to rewrite the draft answer based on critic feedback and search results
async def _run_corrector(
    question: str,
    draft_answer: str,
    search_results: str,
    critique: CriticReport,
    ctx: Context,
) -> str:
    """
    Run the Corrector LLM via MCP Sampling to rewrite the draft answer.

    Args:
        question:       The user's original question.
        draft_answer:   The agent's current draft answer.
        search_results: Raw retrieved context (from CRAG resource).
        critique:       The critic's evaluation report.
        ctx:            FastMCP context for logging and sampling.

    Returns:
        The corrected answer generated by the LLM.
    """
    logger.info("Running corrector generation via MCP Sampling...")
    await ctx.info("Running corrector generation via MCP Sampling...")
    problems = critique.problems

    corrector_messages = [
        SamplingMessage(
            role="assistant",
            content=TextContent(
                type="text",
                text=(
                    "You are a correction editor.\n"
                    "Rewrite answers using ONLY the supplied search results."
                ),
            ),
        ),
        SamplingMessage(
            role="user",
            content=TextContent(
                type="text",
                text=(f"""
                    QUESTION:
                    {question}

                    SEARCH RESULTS:
                    {search_results}

                    CURRENT ANSWER:
                    {draft_answer}

                    PROBLEMS IDENTIFIED:
                    {json.dumps(problems, indent=2)}

                    Instructions:

                    - Fix every problem.
                    - Remove unsupported claims.
                    - Add omitted information.
                    - Keep answer concise.
                    - Use only search results.
                    - If the search results do not support a conclusion,
                    explicitly state that the evidence is insufficient.
                    - Do not infer, speculate, recommend, or generalize.
                    - Every statement must be traceable to the search results.

                    Output only the corrected answer.
                    """),
            ),
        ),
    ]

    result = await ctx.sample(
        messages=corrector_messages,
        max_tokens=2000,
        temperature=0.1,
    )

    if hasattr(result, "text"):
        return result.text

    return str(result)


# 6. @resource — Hierarchical CRAG knowledge base


@mcp.resource("knowledge://domain/docs/{query}")
async def domain_knowledge_resource(query: str, ctx: Context) -> str:
    """
    Hierarchical CRAG knowledge resource with LLM-based ToT evaluation.

    Implements the enhanced CRAG pipeline:
      1. Multi-query expansion
      2. Hierarchical 3-level retrieval (domain → section → sentence)
      3. Tree-of-Thought LLM evaluation (3-persona assessment)
      4. Tavily web fallback when internal docs are insufficient

    Args:
        query: The user's question or research topic.
        ctx:   FastMCP context for logging and MCP protocol operations.

    Returns:
        Formatted context string with retrieved and evaluated knowledge.
    """
    logger.info("domain_knowledge_resource called | query='%s'", query)
    await ctx.info(f"CRAG resource invoked for query: '{query}'")

    # Step 1: Multi-query expansion with hierarchical retrieval
    await ctx.debug("Step 1: Expanding query and retrieving via hierarchy...")
    retrieved = _multi_query_retrieve_hierarchical(query, ctx)
    await ctx.info(
        f"Step 1 complete — hierarchical retrieval found " f"{len(retrieved)} chunks."
    )

    # Step 2: LLM-based ToT evaluation
    await ctx.debug("Step 2: Running Tree-of-Thought LLM evaluation (3 personas)...")
    selected_chunks, fallback_needed = await _tot_evaluate_with_llm(
        query, retrieved, ctx
    )
    await ctx.info(
        f"Step 2 complete — {len(selected_chunks)} chunks passed ToT | "
        f"fallback_needed={fallback_needed}"
    )

    results: list[str] = []

    # Step 3: Format internal results
    if selected_chunks:
        await ctx.debug("Step 3: Formatting selected internal chunks...")
        for chunk in selected_chunks:
            path = chunk["domain"]
            if chunk["section"]:
                path += f" / {chunk['section']}"
            results.append(f"[{chunk['level'].upper()} | {path}]\n{chunk['text']}")

    # Step 4: Tavily fallback
    if fallback_needed:
        await ctx.info(
            "Step 4: Internal docs insufficient - triggering Tavily fallback..."
        )
        tavily_results = await _tavily_fallback(query, ctx)
        if tavily_results:
            results.append("\n--- Web Fallback (Tavily) ---")
            results.extend(tavily_results)
            await ctx.info(
                f"Step 4 complete — added {len(tavily_results)} Tavily results."
            )
        else:
            await ctx.warning("Tavily fallback returned no results.")

    if not results:
        return (
            "No relevant context found in the internal knowledge base or "
            "web fallback for this query."
        )

    formatted = "\n\n".join(results)
    await ctx.info("CRAG pipeline complete - returning context to agent.")
    return formatted


# 7. @tool — Reflection via MCP Sampling (Critic + Corrector loop)
@mcp.tool()
async def reflect_and_correct(
    question: str,
    draft_answer: str,
    search_results: str,
    ctx: Context,
) -> str:
    """
    Reflection Tool using MCP Sampling.

    Workflow:

    Critic
      ↓
    If sufficient -> return answer

    Else:
      ↓
    Corrector
      ↓
    Critic again
      ↓
    Repeat until:
      - sufficient OR
      - max iterations reached
    """

    logger.info(
        "reflect_and_correct invoked | question='%s'",
        question[:80],
    )

    await ctx.info(f"Reflection tool invoked for question: '{question[:80]}'")

    MAX_ITERATIONS = 2

    current_answer = draft_answer

    retry_count = 0

    final_critique = None

    # Reflection Loop

    for iteration in range(MAX_ITERATIONS):

        await ctx.info(f"Reflection iteration " f"{iteration + 1}/{MAX_ITERATIONS}")

        logger.info(
            "Reflection iteration %d/%d",
            iteration + 1,
            MAX_ITERATIONS,
        )

        # Stage 1: Critic

        try:
            await ctx.info(f"Running CRITIC pass " f"{iteration + 1}/{MAX_ITERATIONS}")

            critique = await _run_critic(
                question=question,
                draft_answer=current_answer,
                search_results=search_results,
                ctx=ctx,
            )

            final_critique = critique

            logger.info(
                "Critic result | sufficient=%s | problems=%d",
                critique.is_sufficient,
                len(critique.problems),
            )

        except Exception as exc:
            logger.error(
                "Critic failed on iteration %d: %s",
                iteration + 1,
                exc,
            )

            await ctx.error(f"Critic failed on iteration " f"{iteration + 1}: {exc}")

            result = {
                "critique": str(exc),
                "corrected_answer": current_answer,
                "is_sufficient": False,
                "retry_count": retry_count,
            }

            return json.dumps(result, indent=2)

        # Critic says answer is sufficient

        if critique.is_sufficient:

            logger.info(
                "Critic marked answer sufficient " "at iteration %d",
                iteration + 1,
            )

            await ctx.info("Answer deemed sufficient. " "Skipping correction.")

            break

        # Stage 2: Corrector

        await ctx.info(f"Running CORRECTOR pass " f"{iteration + 1}/{MAX_ITERATIONS}")

        logger.info(
            "Running corrector | problems=%d",
            len(critique.problems),
        )

        try:

            corrected_answer = await _run_corrector(
                question=question,
                draft_answer=current_answer,
                search_results=search_results,
                critique=critique,
                ctx=ctx,
            )

            current_answer = corrected_answer

            retry_count += 1

            logger.info(
                "Corrector completed | retry_count=%d",
                retry_count,
            )

        except Exception as exc:

            logger.error(
                "Corrector failed on iteration %d: %s",
                iteration + 1,
                exc,
            )

            await ctx.error(f"Corrector failed on iteration " f"{iteration + 1}: {exc}")

            break

    # Safety fallback

    if final_critique is None:

        result = {
            "critique": "No critique generated.",
            "corrected_answer": current_answer,
            "is_sufficient": False,
            "retry_count": retry_count,
        }

        return json.dumps(result, indent=2)

    # Final result

    result = {
        "critique": final_critique.model_dump(),
        "corrected_answer": current_answer,
        "is_sufficient": final_critique.is_sufficient,
        "retry_count": retry_count,
    }

    logger.info(
        "Reflection complete | sufficient=%s | retries=%d",
        final_critique.is_sufficient,
        retry_count,
    )

    await ctx.info(
        f"Reflection complete | "
        f"is_sufficient={final_critique.is_sufficient} | "
        f"retries={retry_count}"
    )

    return json.dumps(result, indent=2)


# 8. Entry point to start the FastMCP server


def main() -> None:
    """Start the FastMCP server over streamable-http transport."""
    logger.info("Starting ThinkingAgent MCP Server on http://localhost:8080/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080, path="/mcp")


if __name__ == "__main__":
    main()
