from src.state import AgentState


def handle_zero_results(state: AgentState) -> dict:
    """Handle cases where we couldn't find a paper."""

    error = state.get("error")

    return {
        "error": error or "No matching arXiv paper was found."
    }


def handle_parse_failure(state: AgentState) -> dict:
    """Handle PDF download or text extraction failures."""

    return {
        "error": state.get(
            "error",
            "The paper could not be parsed successfully.",
        )
    }


def handle_embedding_failure(state: AgentState) -> dict:
    """Handle failures while creating or storing embeddings."""

    return {
        "error": state.get(
            "error",
            "The paper could not be embedded or stored.",
        )
    }