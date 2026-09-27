from src.nodes.failures import (
    handle_embedding_failure,
    handle_parse_failure,
    handle_zero_results,
)


def test_handle_zero_results():
    state = {
        "error": None,
    }

    result = handle_zero_results(state)

    assert result["error"] == "No matching arXiv paper was found."


def test_handle_parse_failure():
    state = {
        "error": "PDF download failed.",
    }

    result = handle_parse_failure(state)

    assert result["error"] == "PDF download failed."


def test_handle_embedding_failure():
    state = {
        "error": "Embedding model failed.",
    }

    result = handle_embedding_failure(state)

    assert result["error"] == "Embedding model failed."