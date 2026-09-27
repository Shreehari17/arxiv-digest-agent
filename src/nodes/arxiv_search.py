import logging

import arxiv

from src.state import AgentState, PaperMeta


logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 10


def _result_to_paper_meta(result: "arxiv.Result") -> PaperMeta:
    """Convert an arXiv API result into our own PaperMeta object."""

    return PaperMeta(
        arxiv_id=result.get_short_id(),
        title=result.title.strip(),
        authors=[author.name for author in result.authors],
        abstract=result.summary.strip(),
        pdf_url=result.pdf_url,
        categories=result.categories,
        published=result.published.date().isoformat(),
    )


def arxiv_search(state: AgentState) -> dict:
    """Search arXiv using the user's topic."""

    query = state["parsed_query"]

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=_MAX_CANDIDATES,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    try:
        results = list(client.results(search))

    except Exception as exc:
        logger.warning(
            "arXiv search failed for query %r: %s",
            query,
            exc,
        )

        return {
            "candidates": [],
            "error": f"arXiv search failed: {exc}",
        }

    candidates = [
        _result_to_paper_meta(result)
        for result in results
    ]

    return {
        "candidates": candidates
    }


def fetch_by_id(state: AgentState) -> dict:
    """Fetch one specific paper using its arXiv ID."""

    arxiv_id = state["parsed_query"]

    client = arxiv.Client()

    search = arxiv.Search(
        id_list=[arxiv_id]
    )

    try:
        results = list(client.results(search))

    except Exception as exc:
        logger.warning(
            "arXiv lookup failed for ID %r: %s",
            arxiv_id,
            exc,
        )

        return {
            "error": f"Could not resolve arXiv ID '{arxiv_id}': {exc}"
        }

    if not results:
        return {
            "error": f"No paper found for arXiv ID '{arxiv_id}'."
        }

    paper = _result_to_paper_meta(results[0])

    return {
        "selected_paper": paper
    }


def select_paper(state: AgentState) -> dict:
    """Select the top-ranked paper from the search results."""

    candidates = state["candidates"]

    return {
        "selected_paper": candidates[0]
    }


def route_search_results(state: AgentState) -> str:
    """Choose what to do after a topic search."""

    if not state["candidates"]:
        return "handle_zero_results"

    return "select_paper"


def route_fetch_by_id(state: AgentState) -> str:
    """Choose what to do after fetching a paper by ID."""

    if not state.get("selected_paper"):
        return "handle_zero_results"

    return "fetch_and_parse"