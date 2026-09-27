from dataclasses import dataclass
from typing import Literal, Optional, TypedDict


@dataclass
class PaperMeta:
    """Metadata for a single arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    categories: list[str]
    published: str
@dataclass
class Briefing:
    title: str
    authors: list[str]
    arxiv_id: str
    published: str
    pdf_url: str
    why_it_matters: str
    problem_statement: str
    methods: list[str]
    key_results: list[str]
    limitations: list[str]
    follow_up_questions: list[str]

@dataclass
class QATurn:
    question: str
    answer: str

class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""


    user_input: str
    intent: Optional[Literal["topic", "paper_id"]]
    parsed_query: Optional[str]

    briefing: Optional[Briefing]
    candidates: list[PaperMeta]
    selected_paper: Optional[PaperMeta]
    next_action: str
    
    raw_text: Optional[str]
    parse_ok: bool
    truncated: bool


    vectorstore_ref: Optional[str]
    chunk_count: int
    embed_ok: bool
    retrieved_chunks: list[dict]
    conversation_history: list[QATurn]
    continue_qa: bool
    last_answer: Optional[str]


    error: Optional[str]


def initial_state(user_input: str) -> AgentState:
    """Create the initial state for one graph run."""

    return AgentState(
        user_input=user_input,
        intent=None,
        parsed_query=None,
        candidates=[],
        selected_paper=None,
        raw_text=None,
        parse_ok=False,
        truncated=False,
        vectorstore_ref=None,
        chunk_count=0,
        embed_ok=False,
        retrieved_chunks=[],
        conversation_history=[],
        error=None,
        briefing=None,
        continue_qa=False,
        last_answer=None,
        next_action="qa",
        
    )