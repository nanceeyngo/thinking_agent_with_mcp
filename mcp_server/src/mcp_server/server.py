"""
mcp_server/server.py

FastMCP Server exposing:
  1. @tool  - reflect_and_correct   : Two-stage Reflection via MCP Sampling
                                       (server holds NO LLM - delegates to client)
  2. @resource - knowledge://domain/docs : Hierarchical CRAG with ToT evaluation
                                           and Tavily fallback

Transport: streamable-http  (http://localhost:8080/mcp)

IMPORTANT: This server has NO LLM client, NO LangChain, and NO LangGraph
dependency. Every step that requires language-model reasoning (query
expansion is rule-based and LLM-free; ToT evaluation and the critic/
corrector loop) delegates to the connected MCP client via
`ctx.sample(...)` (MCP Sampling). The server only ever orchestrates;
it never calls an LLM API directly and never imports an LLM SDK.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
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
#
#    LOGGING CONTRACT: every helper function that performs a retrieval or
#    evaluation step takes `ctx` and forwards its key milestones to the
#    client via ctx.debug()/ctx.info()/ctx.warning(), in addition to the
#    local `logger.*` call. The local logger and the ctx forwarder are
#    intentionally kept in lock-step so the server console and the
#    client-side mcp_agent_system.log show the same narrative.

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


# 5. Hierarchical index

#    The structure below is a real 3-level tree. Each level carries an
#    inverted index (word -> set of child keys) built once at startup,
#    so a lookup for "which children contain word W" is an O(1) dict
#    access instead of an O(n) scan. Top-k-per-parent selection is also
#    done independently per parent, so it is mathematically guaranteed
#    rather than approximated.


@dataclass
class SentenceNode:
    domain: str
    section: str
    index: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.domain}::{self.section}::{self.index}"


@dataclass
class SectionNode:
    domain: str
    name: str
    summary: str
    sentences: list[SentenceNode] = field(default_factory=list)
    # word -> set of sentence indices within this section
    sentence_inverted_index: dict[str, set[int]] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.domain}::{self.name}::summary"


@dataclass
class DomainNode:
    name: str
    summary: str
    sections: dict[str, SectionNode] = field(default_factory=dict)
    # word -> set of section names within this domain
    section_inverted_index: dict[str, set[str]] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.name}::summary"


def _tokenize(text: str) -> set[str]:
    """Lowercase whitespace tokeniser shared by index-build and query time."""
    return set(text.lower().split())


def _build_hierarchical_tree() -> tuple[dict[str, DomainNode], dict[str, set[str]]]:
    """
    Build the 3-level tree (domain -> section -> sentence) plus a
    top-level inverted index (word -> set of domain names).

    This replaces the old `_build_hierarchical_index()` flat-list
    builder. Every level gets its own inverted index built once here,
    so retrieval never has to scan a level linearly again.
    """
    domains: dict[str, DomainNode] = {}
    domain_inverted_index: dict[str, set[str]] = {}

    for domain_name, sections in DOMAIN_KNOWLEDGE.items():
        domain_summary = (
            f"Domain: {domain_name}. Sections: {', '.join(sections.keys())}."
        )
        domain_node = DomainNode(name=domain_name, summary=domain_summary)

        # Index the domain summary text into the top-level inverted index
        for word in _tokenize(domain_summary):
            domain_inverted_index.setdefault(word, set()).add(domain_name)

        for section_name, sentences in sections.items():
            section_summary = (
                f"Section '{section_name}' in domain '{domain_name}': "
                + " | ".join(sentences[:2])
            )
            section_node = SectionNode(
                domain=domain_name, name=section_name, summary=section_summary
            )

            # Index the section summary into the domain's section-level index
            for word in _tokenize(section_summary):
                domain_node.section_inverted_index.setdefault(word, set()).add(
                    section_name
                )

            for idx, sentence in enumerate(sentences):
                sentence_node = SentenceNode(
                    domain=domain_name, section=section_name, index=idx, text=sentence
                )
                section_node.sentences.append(sentence_node)
                for word in _tokenize(sentence):
                    section_node.sentence_inverted_index.setdefault(word, set()).add(
                        idx
                    )

            domain_node.sections[section_name] = section_node

        domains[domain_name] = domain_node

    return domains, domain_inverted_index


HIERARCHICAL_TREE, DOMAIN_INVERTED_INDEX = _build_hierarchical_tree()
_total_sentences = sum(
    len(section.sentences)
    for domain in HIERARCHICAL_TREE.values()
    for section in domain.sections.values()
)
logger.info(
    "Hierarchical tree built — %d domains, %d sections, %d sentences.",
    len(HIERARCHICAL_TREE),
    sum(len(d.sections) for d in HIERARCHICAL_TREE.values()),
    _total_sentences,
)


# 6. Internal helpers for CRAG


async def _expand_query(raw_query: str, ctx: Context) -> list[str]:
    """
    Multi-query expansion: generate query variants that stay grounded in
    the original query's own terms.

    The approach here expands the query by substituting/adding
    synonyms and closely related terms for words that are actually in
    the query, so every variant is still anchored to the original
    subject matter. No LLM call is made (the server holds no LLM); this
    is deliberately a deterministic, auditable rule-based expansion -
    only the call site (ctx forwarding) changed from the prior version,
    not the "no LLM" design constraint.

    Args:
        raw_query: The original user query.
        ctx: FastMCP context, used to forward expansion progress to the
            client.

    Returns:
        Ordered, de-duplicated list of query variants, always starting
        with the original raw_query.
    """
    await ctx.debug(f"Query expansion: starting for raw_query='{raw_query}'")

    q_words = raw_query.lower().split()
    variants: list[str] = [raw_query]

    # Synonym/related-term map. Each key is a term that may appear in the
    # query; each value is a short list of closely related terms that
    # extend (not replace) the query's own subject matter.
    related_terms: dict[str, list[str]] = {
        "open": ["open-source", "transparency", "auditable weights"],
        "open-source": ["open weights", "transparency", "community audit"],
        "closed": ["closed-source", "proprietary", "restricted access"],
        "source": ["weights", "training data access"],
        "quantum": ["qubit", "quantum error correction"],
        "qubit": ["quantum processor", "superconducting qubit"],
        "computing": ["processor", "hardware"],
        "regulat": ["policy", "compliance", "government oversight"],
        "regulation": ["policy", "compliance", "government oversight"],
        "govern": ["policy", "regulation", "oversight"],
        "policy": ["regulation", "government approach"],
        "law": ["regulation", "legal compliance"],
        "align": ["alignment technique", "RLHF", "constitutional AI"],
        "alignment": ["RLHF", "constitutional AI", "scalable oversight"],
        "safety": ["alignment", "risk mitigation"],
        "rlhf": ["reinforcement learning from human feedback", "reward modelling"],
        "constitutional": ["constitutional AI", "self-critique"],
    }

    matched_terms: list[str] = []
    for word in q_words:
        # match on prefix so "regulation"/"regulate"/"regulatory" all hit
        # the "regulat" key, etc.
        for key, extensions in related_terms.items():
            if word.startswith(key) or key.startswith(word):
                matched_terms.append(key)
                for ext in extensions:
                    candidate = f"{raw_query} {ext}"
                    variants.append(candidate)
                break  # one match per query word is enough

    if matched_terms:
        await ctx.debug(
            f"Query expansion: matched {len(matched_terms)} term(s) in query: "
            f"{matched_terms}"
        )
    else:
        await ctx.debug(
            "Query expansion: no related-term matches found; "
            "falling back to keyword-only variant."
        )

    # Fallback variant: the query's own significant keywords (len > 4),
    # still entirely derived from the original query, never injected
    # boilerplate.
    keywords = [w for w in raw_query.lower().split() if len(w) > 4]
    if keywords:
        variants.append(" ".join(keywords))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    await ctx.info(
        f"Query expansion complete — produced {len(unique)} variant(s) "
        f"from raw query, all anchored to original terms."
    )
    logger.debug("Query expansion produced %d variants: %s", len(unique), unique)
    return unique


async def _hierarchical_retrieve(
    raw_query: str, ctx: Context, top_k: int = 3
) -> list[dict[str, Any]]:
    """
    True hierarchical retrieval over the tree structure (3-level indexing):

    Level 1: O(1) inverted-index lookup per query word to score domains.
    Level 2: For each selected domain only, O(1) inverted-index lookup
             within that domain's own section index — never touches
             other domains' sections.
    Level 3: For each selected section only, O(1) inverted-index
             lookup within that section's own sentence index — never
             touches other sections' sentences.

    Fix for feedback #4/#5: because each level's inverted index is
    scoped to its parent (e.g. a domain's section_inverted_index only
    contains that domain's sections), top_k selection at level 2 is
    done independently per domain, and top_k selection at level 3 is
    done independently per section — so "top_k per domain/section" is
    a real per-parent guarantee, not an approximation from slicing a
    globally sorted list.

    Returns: List of top sentence-level chunks with full hierarchical context.
    """
    q_words = set(raw_query.lower().split())

    # LEVEL 1: Domain-level search via inverted index (O(words) lookups,
    # not O(n) scan of every domain).
    await ctx.debug(f"Hierarchical retrieval L1: scoring domains for '{raw_query}'")
    domain_scores: dict[str, float] = {}
    for word in q_words:
        for domain_name in DOMAIN_INVERTED_INDEX.get(word, ()):
            domain_scores[domain_name] = domain_scores.get(domain_name, 0) + 1.0

    top_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]
    selected_domain_names = [d[0] for d in top_domains]
    if not selected_domain_names:
        selected_domain_names = list(HIERARCHICAL_TREE.keys())
        await ctx.debug(
            "Hierarchical retrieval L1: no domain matches; "
            f"falling back to all {len(selected_domain_names)} domains."
        )
    else:
        await ctx.info(
            f"Hierarchical retrieval L1 complete — {len(selected_domain_names)} "
            f"relevant domain(s): {selected_domain_names}"
        )
    logger.debug("L1 selected domains: %s", selected_domain_names)

    # LEVEL 2: Section-level search, scoped PER DOMAIN using that
    # domain's own section_inverted_index only.
    await ctx.debug("Hierarchical retrieval L2: scoring sections within top domains")
    selected_sections: list[tuple[str, str]] = []  # (domain, section)
    for domain_name in selected_domain_names:
        domain_node = HIERARCHICAL_TREE[domain_name]
        section_scores: dict[str, float] = {}
        for word in q_words:
            for section_name in domain_node.section_inverted_index.get(word, ()):
                section_scores[section_name] = section_scores.get(section_name, 0) + 1.0

        if section_scores:
            top_sections_in_domain = sorted(
                section_scores.items(), key=lambda x: x[1], reverse=True
            )[:top_k]
            for section_name, _ in top_sections_in_domain:
                selected_sections.append((domain_name, section_name))
        else:
            # No section-level match in this domain — include all of its
            # sections so a domain-level hit still surfaces content.
            for section_name in domain_node.sections:
                selected_sections.append((domain_name, section_name))

    await ctx.info(
        f"Hierarchical retrieval L2 complete — {len(selected_sections)} "
        f"relevant section(s) across {len(selected_domain_names)} domain(s)."
    )
    logger.debug("L2 selected sections: %s", selected_sections)

    # LEVEL 3: Sentence-level search, scoped PER SECTION using that
    # section's own sentence_inverted_index only.
    await ctx.debug("Hierarchical retrieval L3: scoring sentences within top sections")
    result: list[dict[str, Any]] = []
    for domain_name, section_name in selected_sections:
        section_node = HIERARCHICAL_TREE[domain_name].sections[section_name]
        sentence_scores: dict[int, float] = {}
        for word in q_words:
            for sent_idx in section_node.sentence_inverted_index.get(word, ()):
                sentence_scores[sent_idx] = sentence_scores.get(sent_idx, 0) + 1.0

        top_sentences_in_section = sorted(
            sentence_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        for sent_idx, score in top_sentences_in_section:
            node = section_node.sentences[sent_idx]
            result.append(
                {
                    "level": "sentence",
                    "domain": node.domain,
                    "section": node.section,
                    "text": node.text,
                    "id": node.id,
                    "score": score,
                }
            )

    # Final ranking across the merged per-section top-k results.
    result.sort(key=lambda c: c["score"], reverse=True)
    await ctx.info(
        f"Hierarchical retrieval L3 complete — retrieved {len(result)} "
        f"sentence-level chunk(s)."
    )
    logger.debug("L3 retrieved %d sentence chunks.", len(result))
    return result


async def _tot_evaluate_with_llm(
    query: str,
    chunks: list[dict[str, Any]],
    ctx: Context,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Tree-of-Thought evaluation using LLM via MCP Sampling.

    The LLM evaluates each chunk from three named PERSPECTIVES:
      - relevance:   does this chunk directly help answer the query?
      - accuracy:    is the chunk technically precise and unambiguous,
                     on its own terms (not whether it agrees with any
                     external source — the LLM has no other source here)?
      - coverage:    combined with the other kept chunks, does this one
                     help fill a gap in what's needed to fully answer
                     the query, or is it redundant with chunks already
                     scored highly?

    One consistent term ("perspective") is used throughout, each
    perspective is defined inline, `fallback_needed` has an explicit
    trigger rule, and the reasoning field's scope is explicitly
    constrained to covering both the per-chunk scores and the
    fallback decision.

    Returns: (selected_chunks, fallback_needed)
    """
    if not chunks:
        await ctx.warning("ToT evaluation: no chunks to evaluate — fallback required.")
        return [], True

    await ctx.debug(
        f"ToT evaluation: preparing {len(chunks)} chunk(s) for 3-perspective "
        "LLM evaluation via MCP Sampling."
    )

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
            role="user",
            content=TextContent(
                type="text",
                text=(
                    "You are evaluating retrieved knowledge-base chunks for "
                    "their usefulness in answering a user's query. Score each "
                    "chunk from three PERSPECTIVES, defined below. Use the "
                    "word 'perspective' consistently; do not call these "
                    "'personas'.\n\n"
                    "PERSPECTIVES (score each 0.0-1.0):\n"
                    "- relevance_score: How directly does this chunk help "
                    "answer the QUERY? 1.0 = directly on-topic and "
                    "answers part of the query; 0.0 = unrelated.\n"
                    "- accuracy_score: How precise and unambiguous is the "
                    "chunk's own claim, taken at face value? 1.0 = a "
                    "clear, specific, checkable statement; 0.0 = vague or "
                    "internally contradictory. (You are not fact-checking "
                    "against outside knowledge — judge only the chunk's "
                    "internal clarity and specificity.)\n"
                    "- coverage_score: Combined with the OTHER chunks listed "
                    "(not in isolation), does this chunk add information "
                    "needed to fully answer the query, or is it redundant "
                    "with a higher-scoring chunk already covering the same "
                    "point? 1.0 = fills a real gap; 0.0 = fully redundant.\n\n"
                    f"QUERY: {query}\n\n"
                    f"CHUNKS:\n{chunk_list}\n\n"
                    "FALLBACK RULE: set fallback_needed=true if, after "
                    "scoring, fewer than half of the chunks have an average "
                    "of the three scores >= 0.5, OR if no remaining chunk "
                    "directly addresses the main subject of the QUERY. "
                    "Otherwise set fallback_needed=false.\n\n"
                    "Output ONLY this JSON shape (no extra text):\n"
                    "{\n"
                    '  "evaluations": [{"id": 0, "relevance_score": 0.8, '
                    '"accuracy_score": 0.9, "coverage_score": 0.7}],\n'
                    '  "fallback_needed": false,\n'
                    '  "reasoning": "One or two sentences explaining BOTH '
                    "(a) why the scores were assigned as they were, and "
                    "(b) why fallback_needed was set to that value, per "
                    'the FALLBACK RULE above."\n'
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

        tot_text = tot_result.text if hasattr(tot_result, "text") else str(tot_result)

        try:
            tot_data = json.loads(tot_text)
            evaluations = tot_data.get("evaluations", [])
            fallback_needed = tot_data.get("fallback_needed", True)
            reasoning = tot_data.get("reasoning", "")

            kept_chunk_ids = {
                e["id"]
                for e in evaluations
                if (
                    e.get("relevance_score", 0)
                    + e.get("accuracy_score", 0)
                    + e.get("coverage_score", 0)
                )
                / 3
                >= 0.5
            }

            selected = [c for i, c in enumerate(chunks) if i in kept_chunk_ids]

            logger.info(
                "ToT evaluation complete — kept %d/%d chunks | fallback=%s | "
                "reasoning=%s",
                len(selected),
                len(chunks),
                fallback_needed,
                reasoning[:200],
            )
            await ctx.info(
                f"ToT evaluation complete — kept {len(selected)}/{len(chunks)} "
                f"chunks | fallback_needed={fallback_needed}"
            )
            if reasoning:
                await ctx.debug(f"ToT evaluation reasoning: {reasoning[:300]}")

            return selected, fallback_needed
        except (json.JSONDecodeError, KeyError) as parse_exc:
            logger.warning(
                "ToT LLM response not valid JSON (%s), using all chunks", parse_exc
            )
            await ctx.warning(
                "ToT evaluation: LLM response was not valid JSON — "
                "keeping all chunks and forcing fallback."
            )
            return chunks, True

    except Exception as exc:
        logger.error("ToT LLM evaluation failed: %s", exc)
        await ctx.error(f"ToT evaluation failed: {exc}")
        return chunks, True


async def _multi_query_retrieve_hierarchical(
    raw_query: str, ctx: Context
) -> list[dict[str, Any]]:
    """
    Multi-query expansion with hierarchical retrieval for each variant.
    """
    variants = await _expand_query(raw_query, ctx)
    await ctx.debug(
        f"Multi-query retrieval: running hierarchical search for "
        f"{len(variants)} variant(s)."
    )

    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []

    for variant in variants:
        results = await _hierarchical_retrieve(variant, ctx)
        for chunk in results:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                merged.append(chunk)

    await ctx.info(
        f"Multi-query hierarchical retrieval merged {len(merged)} unique "
        f"chunk(s) across {len(variants)} variant(s)."
    )
    logger.debug(
        "Multi-query hierarchical retrieval merged %d unique chunks.", len(merged)
    )
    return merged


async def _tavily_fallback(query: str, ctx: Context) -> list[str]:
    """
    Run a Tavily web search as fallback when internal docs are insufficient.

    Uses the `tavily-python` SDK directly (no LangChain).
    `tavily-python` is a thin REST client with no
    LLM/agent code in it.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key:
        await ctx.warning("TAVILY_API_KEY not set — skipping web fallback.")
        return []

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_api_key)
        raw = client.search(query=query, max_results=3)

        results = raw.get("results", []) if isinstance(raw, dict) else []
        formatted = [
            f"[WEB | {r.get('url', 'unknown')}]\n{r.get('content', '')}"
            for r in results
            if r.get("content")
        ]
        await ctx.info(f"Tavily fallback returned {len(formatted)} result(s).")
        return formatted

    except Exception as exc:
        logger.error("Tavily fallback error: %s", exc)
        await ctx.error(f"Tavily fallback failed: {exc}")
        return []


async def _run_critic(
    question: str,
    draft_answer: str,
    search_results: str,
    ctx: Context,
) -> CriticReport:
    """
    Run the Critic LLM via MCP Sampling to evaluate the draft answer.

    An explicit task instruction is added below, before
    the schema.

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
            role="user",
            content=TextContent(
                type="text",
                text=(f"""
                    TASK: Critique the DRAFT ANSWER below against the SEARCH
                    RESULTS. Check it for hallucinations (claims not
                    supported by the search results), contradictions (claims
                    that conflict with the search results), and omissions
                    (information present in the search results that is
                    important to answering the QUESTION but missing from the
                    draft). Then decide whether the draft is sufficient as-is.

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

    text = result.text if hasattr(result, "text") else str(result)

    logger.info("Critic response:\n%s", text)
    try:
        data = json.loads(text)

        report = CriticReport(
            problems=data.get("problems", []),
            is_sufficient=data.get("is_sufficient", False),
            reason=data.get("reason", ""),
            evidence=data.get("evidence", []),
        )
        await ctx.info(
            f"Critic evaluation complete — is_sufficient={report.is_sufficient} "
            f"| {len(report.problems)} problem(s) found."
        )
        return report
    except Exception:
        logger.exception("Critic returned invalid JSON")
        await ctx.warning("Critic returned invalid JSON — treating as insufficient.")
        return CriticReport(
            problems=[text],
            is_sufficient=False,
            reason="Critic returned invalid JSON",
            evidence=[],
        )


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

    corrected = result.text if hasattr(result, "text") else str(result)
    await ctx.info(f"Corrector produced a {len(corrected)}-character revision.")
    return corrected


# 7. @resource — Hierarchical CRAG knowledge base


@mcp.resource("knowledge://domain/docs/{query}")
async def domain_knowledge_resource(query: str, ctx: Context) -> str:
    """
    Hierarchical CRAG knowledge resource with LLM-based ToT evaluation.

    Implements the enhanced CRAG pipeline:
      1. Multi-query expansion (rule-based, grounded in the original query)
      2. Hierarchical 3-level tree retrieval (domain → section → sentence)
      3. Tree-of-Thought LLM evaluation (3-perspective assessment, via
         MCP Sampling — the server never calls an LLM directly)
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
    retrieved = await _multi_query_retrieve_hierarchical(query, ctx)
    await ctx.info(
        f"Step 1 complete — hierarchical retrieval found {len(retrieved)} chunks."
    )

    # Step 2: LLM-based ToT evaluation
    await ctx.debug(
        "Step 2: Running Tree-of-Thought LLM evaluation (3 perspectives)..."
    )
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
        await ctx.info(f"Step 3 complete — formatted {len(results)} internal chunk(s).")

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
        await ctx.warning(
            "CRAG pipeline found no usable context from internal docs or web fallback."
        )
        return (
            "No relevant context found in the internal knowledge base or "
            "web fallback for this query."
        )

    formatted = "\n\n".join(results)
    await ctx.info("CRAG pipeline complete - returning context to agent.")
    return formatted


# 8. @tool — Reflection via MCP Sampling (Critic + Corrector loop)
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

        await ctx.info(f"Reflection iteration {iteration + 1}/{MAX_ITERATIONS}")

        logger.info(
            "Reflection iteration %d/%d",
            iteration + 1,
            MAX_ITERATIONS,
        )

        # Stage 1: Critic

        try:
            await ctx.info(f"Running CRITIC pass {iteration + 1}/{MAX_ITERATIONS}")

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

            await ctx.error(f"Critic failed on iteration {iteration + 1}: {exc}")

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
                "Critic marked answer sufficient at iteration %d",
                iteration + 1,
            )

            await ctx.info("Answer deemed sufficient. Skipping correction.")

            break

        # Stage 2: Corrector

        await ctx.info(f"Running CORRECTOR pass {iteration + 1}/{MAX_ITERATIONS}")

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

            await ctx.error(f"Corrector failed on iteration {iteration + 1}: {exc}")

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


# 9. Entry point to start the FastMCP server


def main() -> None:
    """Start the FastMCP server over streamable-http transport."""
    logger.info("Starting ThinkingAgent MCP Server on http://localhost:8080/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080, path="/mcp")


if __name__ == "__main__":
    main()
