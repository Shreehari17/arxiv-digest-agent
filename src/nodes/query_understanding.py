import re
from typing import Optional

from src.state import AgentState


# Matches modern arXiv IDs such as:
# 1706.03762
# 1706.03762v2
_ARXIV_ID_CORE_RE = re.compile(
    r"^(\d{4}\.\d{4,5})(v\d+)?$"
)

# Matches arXiv URLs such as:
# https://arxiv.org/abs/1706.03762
# https://arxiv.org/pdf/1706.03762
# https://arxiv.org/pdf/1706.03762.pdf
_ARXIV_URL_RE = re.compile(
    r"^https?://arxiv\.org/(?:abs|pdf)/"
    r"(\d{4}\.\d{4,5})(v\d+)?(?:\.pdf)?/?$",
    re.IGNORECASE,
)


def _try_extract_arxiv_id(raw: str) -> Optional[str]:
    """Return an arXiv ID if the input is a direct ID or arXiv URL."""

    candidate = raw.strip()

    # Check arXiv URL first
    url_match = _ARXIV_URL_RE.match(candidate)

    if url_match:
        core_id = url_match.group(1)
        version = url_match.group(2)

        return f"{core_id}{version or ''}"

    # Check plain arXiv ID
    id_match = _ARXIV_ID_CORE_RE.match(candidate)

    if id_match:
        core_id = id_match.group(1)
        version = id_match.group(2)

        return f"{core_id}{version or ''}"

    return None


def query_understanding(state: AgentState) -> dict:
    """Determine whether the user entered a paper ID or a topic."""

    user_input = state["user_input"]

    arxiv_id = _try_extract_arxiv_id(user_input)

    if arxiv_id:
        return {
            "intent": "paper_id",
            "parsed_query": arxiv_id,
        }

    return {
        "intent": "topic",
        "parsed_query": user_input.strip(),
    }


def route_intent(state: AgentState) -> str:
    """Choose the next graph node based on the detected intent."""

    if state.get("intent") == "topic":
        return "arxiv_search"

    if state.get("intent") == "paper_id":
        return "fetch_by_id"

    return "handle_zero_results"