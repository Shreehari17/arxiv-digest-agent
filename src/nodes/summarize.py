import json
import logging

from src.llm import call_llm
from src.prompts import BRIEFING_SYSTEM_PROMPT, build_briefing_prompt
from src.state import AgentState, Briefing
from src.vectorstore import query_collection

logger = logging.getLogger(__name__)

_BRIEFING_QUERIES = [
    "What problem does this paper address and why is it important?",
    "What methods architectures frameworks or approaches does this paper propose?",
    "What are the definitions and full names of the main methods, acronyms, and technical terms?",
    "What are the main experimental results and findings?",
    "What limitations assumptions or weaknesses are discussed in the paper?",
]


def _parse_briefing(raw_response: str) -> Briefing:
    data = json.loads(raw_response)

    return Briefing(
        title=data["title"],
        authors=data["authors"],
        arxiv_id=data["arxiv_id"],
        published=data["published"],
        pdf_url=data["pdf_url"],
        why_it_matters=data["why_it_matters"],
        problem_statement=data["problem_statement"],
        methods=data["methods"],
        key_results=data["key_results"],
        limitations=data["limitations"],
        follow_up_questions=data["follow_up_questions"],
    )


def _retrieve_briefing_evidence(
    vectorstore_ref: str,
    k_per_query: int = 2,
) -> list[str]:
    evidence = []
    seen = set()

    for query in _BRIEFING_QUERIES:
        results = query_collection(
            vectorstore_ref,
            query,
            k=k_per_query,
        )

        documents = results.get("documents") or [[]]

        for document in documents[0]:
            if document not in seen:
                seen.add(document)
                evidence.append(document)

    return evidence


def summarize(state: AgentState) -> dict:
    paper = state["selected_paper"]
    vectorstore_ref = state["vectorstore_ref"]


    evidence = _retrieve_briefing_evidence(vectorstore_ref)

    if not evidence:
        return {
            "error": "Could not retrieve enough paper evidence for the briefing."
        }

    evidence_text = "\n\n---\n\n".join(evidence)

    prompt = build_briefing_prompt(
        paper_text=evidence_text,
        paper=paper,
    )

    try:
        raw_response = call_llm(
            prompt,
            system=BRIEFING_SYSTEM_PROMPT,
        )

        briefing = _parse_briefing(raw_response)

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse briefing JSON: %s", exc)
        return {
            "error": f"Failed to parse LLM briefing: {exc}",
        }

    except Exception as exc:
        logger.warning("Briefing generation failed: %s", exc)
        return {
            "error": f"Failed to generate briefing: {exc}",
        }

    return {
        "briefing": briefing,
        }