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

# 4. Internal helpers for CRAG


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


def _keyword_search(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    """
    Simple keyword-overlap retrieval across the hierarchical index.
    Returns top_k chunks ranked by overlap score.
    """
    q_words = set(query.lower().split())
    scored: list[tuple[float, dict[str, Any]]] = []

    for chunk in HIERARCHICAL_INDEX:
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(q_words & chunk_words)
        if overlap > 0:
            # Weight granular sentences higher than summaries
            level_weight = {"sentence": 1.5, "section": 1.0, "domain": 0.5}.get(
                chunk["level"], 1.0
            )
            scored.append((overlap * level_weight, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def _multi_query_retrieve(raw_query: str) -> list[dict[str, Any]]:
    """
    Run keyword retrieval for each query variant and merge results,
    deduplicating by chunk ID.
    """
    variants = _expand_query(raw_query)
    logger.debug(
        "Multi-query expansion produced %d variants: %s", len(variants), variants
    )

    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []

    for variant in variants:
        results = _keyword_search(variant, top_k=4)
        for chunk in results:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                merged.append(chunk)

    logger.debug("Multi-query retrieval merged %d unique chunks.", len(merged))
    return merged


# 5. Tree-of-Thought evaluation (server-side, LLM-free)
#    Three scoring personas evaluate each chunk; the judge selects the best set.
#    Because the server has no LLM, ToT is implemented as rule-based scoring.


class ToTEvaluation(BaseModel):
    chunk_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    persona_scores: dict[str, float]
    verdict: str  # "keep" | "drop" | "fallback_needed"


def _tot_evaluate_chunks(
    query: str,
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """
    Three-persona ToT evaluation for retrieved chunks.

    Persona A - Analytical: rewards domain/section match and sentence-level detail.
    Persona B - Relevance:  rewards keyword overlap with the raw query.
    Persona C - Coverage:   penalises if only summary-level chunks remain.

    Returns (selected_chunks, fallback_needed).
    fallback_needed = True when average score < 0.35 (internal docs insufficient).
    """
    q_words = set(query.lower().split())
    evaluations: list[ToTEvaluation] = []

    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split())
        overlap_ratio = len(q_words & chunk_words) / max(len(q_words), 1)

        # Persona A — Analytical
        score_a = 0.5 if chunk["level"] == "sentence" else 0.25
        score_a += overlap_ratio * 0.5

        # Persona B — Relevance
        score_b = overlap_ratio

        # Persona C — Coverage (favours sentence-level chunks)
        score_c = {"sentence": 0.8, "section": 0.5, "domain": 0.2}.get(
            chunk["level"], 0.3
        )

        avg_score = (score_a + score_b + score_c) / 3

        verdict = "keep" if avg_score >= 0.35 else "drop"

        eval_result = ToTEvaluation(
            chunk_id=chunk["id"],
            relevance_score=round(avg_score, 3),
            persona_scores={
                "analytical": round(score_a, 3),
                "relevance": round(score_b, 3),
                "coverage": round(score_c, 3),
            },
            verdict=verdict,
        )
        evaluations.append(eval_result)

        logger.debug(
            "ToT eval — chunk='%s' | analytical=%.2f | relevance=%.2f | "
            "coverage=%.2f | avg=%.2f | verdict=%s",
            chunk["id"],
            score_a,
            score_b,
            score_c,
            avg_score,
            verdict,
        )

    # Select kept chunks
    kept_ids = {e.chunk_id for e in evaluations if e.verdict == "keep"}
    selected = [c for c in chunks if c["id"] in kept_ids]

    # Fallback if fewer than 2 chunks survive or average score is very low
    avg_overall = (
        sum(e.relevance_score for e in evaluations) / len(evaluations)
        if evaluations
        else 0.0
    )
    fallback_needed = len(selected) < 2 or avg_overall < 0.35

    logger.info(
        "ToT evaluation complete — kept %d/%d chunks | avg_score=%.3f | "
        "fallback_needed=%s",
        len(selected),
        len(chunks),
        avg_overall,
        fallback_needed,
    )

    return selected, fallback_needed


# 6. @resource — Hierarchical CRAG knowledge base


@mcp.resource("knowledge://domain/docs/{query}")
async def domain_knowledge_resource(query: str, ctx: Context) -> str:
    """
    Hierarchical CRAG knowledge resource.

    Implements the full CRAG pipeline:
      1. Multi-query expansion
      2. Hierarchical retrieval across domain/section/sentence levels
      3. Tree-of-Thought 3-persona evaluation
      4. Tavily web fallback when internal docs are insufficient

    Args:
        query: The user's question or research topic.
        ctx:   FastMCP context for logging and MCP protocol operations.

    Returns:
        Formatted context string with retrieved and evaluated knowledge.
    """
    logger.info("domain_knowledge_resource called | query='%s'", query)
    await ctx.info(f"CRAG resource invoked for query: '{query}'")

    # Step 1: Multi-query expansion
    await ctx.debug("Step 1: Expanding query into semantic variants...")

    retrieved = _multi_query_retrieve(query)
    await ctx.info(f"Step 1 complete — retrieved {len(retrieved)} unique chunks.")

    # Step 2: ToT evaluation
    await ctx.debug("Step 2: Running Tree-of-Thought evaluation (3 personas)...")
    selected_chunks, fallback_needed = _tot_evaluate_chunks(query, retrieved)
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
            "Step 4: Internal docs insufficient — triggering Tavily falllback..."
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
    await ctx.info("CRAG pipeline complete — returning context to agent.")
    return formatted


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


# 7. @tool — Reflection via MCP Sampling
#    CRITICAL: server holds NO LLM. Both critic and corrector calls are
#    delegated to the client's LLM via ctx.request_sampling().
#    Content blocks MUST be arrays of JSON objects, not raw strings.


@mcp.tool()
async def reflect_and_correct(
    question: str,
    draft_answer: str,
    search_results: str,
    ctx: Context,
) -> str:
    """
    Two-stage Reflection Tool implemented via MCP Sampling.

    The server itself holds NO LLM and NO API keys.
    Both the Critic and the Corrector LLM calls are delegated to the
    MCP client via ctx.request_sampling(). The client's locally configured
    LLM executes the generation and returns the text back to the server.

    Stage 1 - Critic: Compares draft against search_results to identify
      hallucinations, contradictions, and omissions.

    Stage 2 - Corrector: Uses the critique + search_results to rewrite
      the draft, grounding all claims in the actual evidence.

    Args:
        question:       The user's original question.
        draft_answer:   The agent's current draft answer.
        search_results: Raw retrieved context (from query_knowledge_base or Tavily).
        ctx:            FastMCP context — used to invoke MCP Sampling.

    Returns:
        JSON string with keys: critique, corrected_answer, is_sufficient.
    """
    logger.info("reflect_and_correct invoked | question='%s'", question[:80])
    await ctx.info(f"Reflection tool invoked for question: '{question[:80]}'")

    # Stage 1: Critic via MCP Sampling
    await ctx.info("Stage 1: Requesting CRITIC generation via MCP Sampling...")

    critic_text = (
        "You are a rigorous Critic. Compare this draft answer against the "
        "search results and identify ALL problems:\n"
        "- Hallucinations: claims in the draft NOT supported by search results\n"
        "- Contradictions: claims that CONFLICT with search results\n"
        "- Omissions: important points in results the draft missed\n\n"
        "Cite specific evidence from search results for each problem.\n"
        "Output ONLY plain-text critique — no JSON, no corrected answer.\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"SEARCH RESULTS (ground truth):\n{search_results}\n\n"
        f"DRAFT ANSWER TO CRITIQUE:\n{draft_answer}"
    )
    try:
        critic_result = await ctx.sample(
            critic_text,
            temperature=0.0,
            max_tokens=1024,
        )
        logger.info(type(critic_result))
        logger.info(repr(critic_result))
        critique = (
            critic_result.text if hasattr(critic_result, "text") else str(critic_result)
        )
        logger.debug("Stage 1 critique received (%d chars).", len(critique))
        await ctx.info("Stage 1 complete — critique received from client LLM.")
    except Exception as exc:
        logger.error("Stage 1 sampling failed: %s", exc)
        await ctx.error(f"Critic sampling failed: {exc}")
        critique = f"Critic unavailable: {exc}"

    # Stage 2: Corrector via MCP Sampling
    await ctx.info("Stage 2: Requesting CORRECTOR generation via MCP Sampling...")

    corrector_text = (
        "You are a skilled Editor. Rewrite the draft answer to"
        "fix EVERY problem"
        "identified in the critique. Use the search results as "
        "your ONLY ground truth.\n\n"
        "Rules:\n"
        "- Replace hallucinated claims with what search results actually say\n"
        "- Do not introduce new claims not in the search results\n"
        "- Address every point in the critique\n"
        "- Output ONLY the corrected answer as prose — no JSON, no labels\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"SEARCH RESULTS (ground truth):\n{search_results}\n\n"
        f"DRAFT ANSWER:\n{draft_answer}\n\n"
        f"CRITIQUE TO FIX:\n{critique}"
    )
    try:
        corrector_result = await ctx.sample(
            corrector_text,
            temperature=0.1,
            max_tokens=2048,
        )
        corrected_answer = (
            corrector_result.text
            if hasattr(corrector_result, "text")
            else str(corrector_result)
        )
        logger.debug(
            "Stage 2 corrected answer received (%d chars).", len(corrected_answer)
        )
        await ctx.info("Stage 2 complete — corrected answer received from client LLM.")
    except Exception as exc:
        logger.error("Stage 2 sampling failed: %s", exc)
        await ctx.error(f"Corrector sampling failed: {exc}")
        corrected_answer = draft_answer  # Fall back to original draft

    # Determine sufficiency
    insufficient_signals = [
        "hallucin",
        "unsupported",
        "speculative",
        "not mentioned",
        "contradicts",
        "no evidence",
        "not grounded",
        "inaccurate",
    ]
    is_sufficient = not any(s in critique.lower() for s in insufficient_signals)

    result = {
        "critique": critique,
        "corrected_answer": corrected_answer,
        "is_sufficient": is_sufficient,
    }

    logger.info("reflect_and_correct complete | is_sufficient=%s", is_sufficient)
    await ctx.info(f"Reflection complete | is_sufficient={is_sufficient}")

    return json.dumps(result, indent=2)


# 8. Entry point


def main() -> None:
    """Start the FastMCP server over streamable-http transport."""
    logger.info("Starting ThinkingAgent MCP Server on http://localhost:8080/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080, path="/mcp")


if __name__ == "__main__":
    main()
