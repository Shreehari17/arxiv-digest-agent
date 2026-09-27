from langgraph.graph import END, START, StateGraph

from src.state import AgentState

from src.nodes.summarize import summarize
from src.nodes.qa import qa_loop, route_qa

from src.nodes.query_understanding import (
    query_understanding,
    route_intent,
)

from src.nodes.arxiv_search import (
    arxiv_search,
    fetch_by_id,
    select_paper,
    route_search_results,
    route_fetch_by_id,
)

from src.nodes.parsing import (
    fetch_and_parse,
    route_parse,
)

from src.nodes.retrieval import (
    chunk_and_embed,
    route_embed,
)

from src.nodes.failures import (
    handle_zero_results,
    handle_parse_failure,
    handle_embedding_failure,
)


def build_graph():
    """Build and compile the research paper agent graph."""

    builder = StateGraph(AgentState)

    builder.add_node("query_understanding", query_understanding)
    builder.add_node("arxiv_search", arxiv_search)
    builder.add_node("select_paper", select_paper)
    builder.add_node("fetch_by_id", fetch_by_id)
    builder.add_node("fetch_and_parse", fetch_and_parse)
    builder.add_node("chunk_and_embed", chunk_and_embed)

    builder.add_node(
        "handle_zero_results",
        handle_zero_results,
    )
    builder.add_node(
        "handle_parse_failure",
        handle_parse_failure,
    )
    builder.add_node(
        "handle_embedding_failure",
        handle_embedding_failure,
    )

    builder.add_edge(
        START,
        "query_understanding",
    )

    builder.add_conditional_edges(
        "query_understanding",
        route_intent,
        {
            "arxiv_search": "arxiv_search",
            "fetch_by_id": "fetch_by_id",
            "handle_zero_results": "handle_zero_results",
        },
    )

    builder.add_conditional_edges(
        "arxiv_search",
        route_search_results,
        {
            "select_paper": "select_paper",
            "handle_zero_results": "handle_zero_results",
        },
    )

    builder.add_edge(
        "select_paper",
        "fetch_and_parse",
    )

    builder.add_conditional_edges(
        "fetch_by_id",
        route_fetch_by_id,
        {
            "fetch_and_parse": "fetch_and_parse",
            "handle_zero_results": "handle_zero_results",
        },
    )

    builder.add_conditional_edges(
        "fetch_and_parse",
        route_parse,
        {
            "parse_ok": "chunk_and_embed",
            "handle_parse_failure": "handle_parse_failure",
        },
    )

    builder.add_conditional_edges(
        "chunk_and_embed",
        route_embed,
        {
            "summarize": "summarize",
            "handle_embedding_failure": "handle_embedding_failure",
        },
    )

    builder.add_node("summarize", summarize)
    builder.add_node("qa_loop", qa_loop)

    builder.add_edge(
        "summarize",
        "qa_loop",
    )

    builder.add_conditional_edges(
        "qa_loop",
        route_qa,
        {
            "qa_loop": "qa_loop",
            "change_paper": "query_understanding",
            "end": END,
        },
    )

    builder.add_edge(
        "handle_embedding_failure",
        END,
    )

    builder.add_edge(
        "handle_zero_results",
        END,
    )

    builder.add_edge(
        "handle_parse_failure",
        END,
    )

    return builder.compile()