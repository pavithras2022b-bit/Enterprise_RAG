"""
Module 12 - LLM Response Generation: prompt construction and citation
extraction.

Provides build_prompt(...), extract_citations(...), and the
PromptBuilder class per the Phase 4 module contract.

extract_citations lives here (not in client.py) rather than as a
separate file, because PromptBuilder defines the citation format
([source_id] markers) and extract_citations parses that exact format
back out -- co-locating them means the format can never silently drift
out of sync between where it's specified and where it's read, the
same lesson already applied in knowledge_graph/cypher_queries.py
(labels derived from the same enum the graph builder writes, rather
than independently hardcoded).

Reference: Capstone Project Guide, Section 3M ("LLM Generation"),
Phase 4 Module 12 contract, Phase 5 file structure
(llm/prompt_builder.py).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from reranking.schema import RankedCandidate

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_INSTRUCTION = (
    "Using only the context below, answer the question directly and concisely. "
    "Each piece of context is labeled with an exact citation id such as "
    "[pdf_ff8f53d904329460_chunk_0000]. "
    "Only state facts that are directly supported by the provided context. "

    "ANSWER STYLE: Give the shortest complete answer that directly satisfies the "
    "question. For 'Who' questions, return the person or department and only the "
    "minimum identifying information needed. For 'When' questions, return the "
    "date, month, period, or timing directly. For questions asking for a name, "
    "system, policy, tool, or entity, return that entity directly. For descriptive "
    "'What', 'Why', or 'How' questions, use one concise sentence unless more detail "
    "is necessary. Do not repeat the question or add extra titles, locations, "
    "examples, explanations, or background unless they are required to answer it. "

    "CITATION REQUIREMENT: Every factual answer MUST include at least one citation. "
    "After each factual statement, copy the exact bracketed citation id from the "
    "context that supports that statement. Never invent, shorten, modify, or omit "
    "a citation id. If you answer with a factual statement but provide no valid "
    "citation, the answer is invalid. "
    "Example format: Project Titan uses ServiceNow and Coupa "
    "[pdf_ff8f53d904329460_chunk_0000]. "
    "A citation by itself is never a valid answer. "

    "If the context does not contain enough information to answer the question, "
    "reply with a short explicit statement that the information is not specified "
    "in the provided context. If a retrieved source directly supports that "
    "insufficiency or omission, cite that source; otherwise do not include a citation. "

    "Do not guess, infer missing facts, or answer using unrelated context."
)

# Matches a bracketed citation marker, e.g. "[doc1_chunk_0003]" or
# "[asha rao|belongs_to|finance]" (a GraphFact's fact_id, which itself
# contains no brackets, so this pattern is safe against both id shapes
# Module 9/11 can hand this module).
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


class PromptBuilderError(Exception):
    """Raised when a prompt cannot be built for the given input."""


class PromptBuilder:
    """
    Builds the grounded, citation-instructed prompt handed to the LLM,
    per Section 3M's template: query + retrieved evidence + an explicit
    instruction to answer only from the provided context and cite
    sources.
    """

    def __init__(self, system_instruction: Optional[str] = None) -> None:
        """
        Args:
            system_instruction: Overrides the default grounding/citation
                instruction. Must still direct the model to cite sources
                using the same "[source_id]" bracket format
                extract_citations() parses, or citation extraction will
                silently find nothing.
        """
        self._system_instruction = system_instruction or _DEFAULT_SYSTEM_INSTRUCTION

    def build_prompt(self, query: str, evidence: list["RankedCandidate"]) -> str:
        """
        Args:
            query: The user's question.
            evidence: Top-N re-ranked candidates from Module 11, in
                the order they should be presented (typically already
                rank-sorted).

        Returns:
            The complete prompt string: system instruction, an
            evidence block with each item's text prefixed by its
            bracketed candidate_id, and the question.

        Raises:
            PromptBuilderError: If query is empty/whitespace-only.
        """
        if not query or not query.strip():
            raise PromptBuilderError("build_prompt() requires a non-empty query.")

        if not evidence:
            evidence_block = "(No relevant evidence was found for this query.)"
        else:
            evidence_block = "\n\n".join(
                f"[{item.candidate_id}] {item.text}" for item in evidence
            )

        return (
            f"{self._system_instruction}\n\n"
            f"Context:\n{evidence_block}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )


def build_prompt(query: str, evidence: list["RankedCandidate"]) -> str:
    """
    Module-level convenience wrapper around PromptBuilder.build_prompt(),
    matching the Phase 4 module contract's function signature
    (build_prompt(query, evidence) -> str). Constructs a default-config
    builder for one-off use.

    Args:
        query: The user's question.
        evidence: Top-N re-ranked candidates from Module 11.

    Returns:
        The complete prompt string.
    """
    return PromptBuilder().build_prompt(query, evidence)


def extract_citations(answer_text: str, evidence: list["RankedCandidate"]) -> list[str]:
    """
    Extract the source ids the generated answer actually cites, per
    the Phase 4 module contract's function signature
    (extract_citations(answer, evidence) -> list[str]).

    Only ids that both (a) appear as a bracketed marker in answer_text
    and (b) correspond to a candidate_id actually present in evidence
    are returned -- this is what makes citations *validated*, not
    merely detected: a hallucinated citation to an id never supplied
    in the prompt's context is filtered out rather than trusted,
    directly serving Section 13.1's citation-accuracy evaluation
    criterion (does the citation actually exist / point to real
    evidence).

    Args:
        answer_text: The LLM's generated answer.
        evidence: The evidence list the prompt was built from
            (typically the same list passed to build_prompt()).

    Returns:
        Cited source ids, in first-appearance order, deduplicated.
        Empty list if answer_text is empty or cites nothing valid.
    """
    if not answer_text:
        return []

    valid_ids = {item.candidate_id for item in evidence}
    seen: set[str] = set()
    citations: list[str] = []

    for match in _CITATION_PATTERN.finditer(answer_text):
        candidate_id = match.group(1).strip()
        if candidate_id in valid_ids and candidate_id not in seen:
            citations.append(candidate_id)
            seen.add(candidate_id)

    return citations
