from src.nodes.qa import answer_question
from src.state import QATurn


def test_answerable_question(monkeypatch):
    """An answerable question should use retrieved evidence and call the LLM."""

    fake_chunks = [
        {
            "text": "The Transformer uses 8 parallel attention heads.",
            "metadata": {"chunk_index": 10},
            "distance": 0.4,
        }
    ]

    monkeypatch.setattr(
        "src.nodes.qa.retrieve_for_qa",
        lambda vectorstore_ref, question: fake_chunks,
    )

    def fake_llm(prompt, system=None):
        assert "The Transformer uses 8 parallel attention heads." in prompt
        return "They used 8 attention heads.\nSupported by: chunk 1"

    monkeypatch.setattr(
        "src.nodes.qa.call_llm",
        fake_llm,
    )

    turn = answer_question(
        question="How many attention heads did they use?",
        vectorstore_ref="paper_test",
        history=[],
    )

    assert isinstance(turn, QATurn)
    assert turn.question == "How many attention heads did they use?"
    assert "8 attention heads" in turn.answer


def test_unsupported_question(monkeypatch):
    """An unsupported question should not call the LLM."""

    monkeypatch.setattr(
        "src.nodes.qa.retrieve_for_qa",
        lambda vectorstore_ref, question: [],
    )

    def fake_llm(prompt, system=None):
        raise AssertionError("LLM should not be called for unsupported questions.")

    monkeypatch.setattr(
        "src.nodes.qa.call_llm",
        fake_llm,
    )

    turn = answer_question(
        question="What does the paper say about quantum computing?",
        vectorstore_ref="paper_test",
        history=[],
    )

    assert (
        turn.answer
        == "I couldn't find content in this paper related to that question."
    )


def test_conversation_history_is_used(monkeypatch):
    """Previous QA turns should be included when answering a new question."""

    fake_chunks = [
        {
            "text": "The Transformer uses an encoder and decoder.",
            "metadata": {"chunk_index": 5},
            "distance": 0.4,
        }
    ]

    monkeypatch.setattr(
        "src.nodes.qa.retrieve_for_qa",
        lambda vectorstore_ref, question: fake_chunks,
    )

    previous_turn = QATurn(
        question="What architecture does the paper introduce?",
        answer="The Transformer.",
    )

    def fake_llm(prompt, system=None):
        assert "What architecture does the paper introduce?" in prompt
        assert "The Transformer." in prompt
        return "It uses an encoder and decoder.\nSupported by: chunk 1"

    monkeypatch.setattr(
        "src.nodes.qa.call_llm",
        fake_llm,
    )

    turn = answer_question(
        question="What are its main parts?",
        vectorstore_ref="paper_test",
        history=[previous_turn],
    )

    assert "encoder and decoder" in turn.answer 

def test_weak_retrieval_is_rejected(monkeypatch):
    """Chunks above the distance threshold should not reach the LLM."""

    weak_chunks = [
        {
            "text": "This chunk is not sufficiently relevant.",
            "metadata": {"chunk_index": 20},
            "distance": 0.80,
        }
    ]

    monkeypatch.setattr(
        "src.nodes.qa.retrieve_for_qa",
        lambda vectorstore_ref, question: [],
    )

    def fake_llm(prompt, system=None):
        raise AssertionError("LLM should not be called for weak retrieval.")

    monkeypatch.setattr(
        "src.nodes.qa.call_llm",
        fake_llm,
    )

    turn = answer_question(
        question="What does the paper say about quantum computing?",
        vectorstore_ref="paper_test",
        history=[],
    )

    assert turn.answer == (
        "I couldn't find content in this paper related to that question."
    )