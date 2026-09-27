from src.nodes.retrieval import chunk_text
from src.vectorstore import store_chunks, query_collection


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_stays_as_one_chunk():
    text = "This is a short paper."

    chunks = chunk_text(text)

    assert chunks == [text]


def test_long_text_is_split_into_multiple_chunks():
    text = "a" * 2500

    chunks = chunk_text(text, chunk_size=1000, overlap=150)

    assert len(chunks) > 1


def test_chunks_have_overlap():
    text = "a" * 2500

    chunks = chunk_text(text, chunk_size=1000, overlap=150)

    assert chunks[0][-150:] == chunks[1][:150]

def test_store_and_query_round_trip():
    collection_name = "test_retrieval_round_trip"

    chunks = [
        "Transformers use self-attention to model relationships between tokens.",
        "Convolutional neural networks are commonly used for image processing.",
        "Reinforcement learning trains agents through rewards and penalties.",
    ]

    metadatas = [
        {"chunk_index": 0},
        {"chunk_index": 1},
        {"chunk_index": 2},
    ]

    ids = ["test_0", "test_1", "test_2"]

    store_chunks(collection_name, chunks, metadatas, ids)

    results = query_collection(
        collection_name,
        "How does self-attention work in Transformers?",
        k=1,
    )

    assert len(results["documents"]) == 1
    assert len(results["documents"][0]) == 1

    retrieved = results["documents"][0][0]

    assert "self-attention" in retrieved

    
def test_qa_retrieval_applies_distance_threshold(monkeypatch):
    """QA retrieval should reject chunks beyond the distance threshold."""

    fake_results = [
        {
            "text": "Relevant chunk",
            "metadata": {"chunk_index": 1},
            "distance": 0.60,
        },
        {
            "text": "Weak chunk",
            "metadata": {"chunk_index": 2},
            "distance": 0.80,
        },
    ]

    monkeypatch.setattr(
        "src.nodes.retrieval.retrieve",
        lambda vectorstore_ref, query, k: fake_results,
    )

    from src.nodes.retrieval import retrieve_for_qa

    results = retrieve_for_qa(
        vectorstore_ref="paper_test",
        question="test question",
    )

    assert len(results) == 1
    assert results[0]["text"] == "Relevant chunk"