from typing import TypedDict
import re
from src.vectorstore import (
    collection_exists_and_populated,
    collection_name_for,
    get_or_create_collection,
    query_collection,
    store_chunks,
)


_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 150
_MIN_CHUNK_CHARS = 50

_DEFAULT_K = 4


_QA_K = 8
_QA_DISTANCE_THRESHOLD = 0.65


class RetrievedChunk(TypedDict):
    text: str
    metadata: dict
    distance: float



_STOPWORDS = {
    "what",
    "what's",
    "which",
    "where",
    "when",
    "who",
    "why",
    "how",
    "does",
    "do",
    "did",
    "is",
    "are",
    "was",
    "were",
    "the",
    "this",
    "that",
    "these",
    "those",
    "from",
    "with",
    "about",
    "into",
    "than",
    "their",
    "they",
    "them",
    "paper",
}


def _extract_lexical_terms(question: str) -> list[str]:
    """
    Extract useful technical terms from a question.

    Keeps hyphenated terms such as 'self-attention' intact and
    normalizes terms for case-insensitive matching.
    """

    words = re.findall(
        r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*",
        question,
    )

    terms = []

    for word in words:
        normalized = word.lower()

        if normalized in _STOPWORDS:
            continue

        if len(normalized) < 4:
            continue

        if normalized not in terms:
            terms.append(normalized)

    return terms


def chunk_text(
    text: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap: int = _CHUNK_OVERLAP,
) -> list[str]:
    """
    Split paper text into overlapping chunks.

    Short documents are kept as a single chunk.
    Very small trailing fragments are discarded.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if len(chunk) >= _MIN_CHUNK_CHARS:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_and_embed(state) -> dict:
    """Chunk the parsed paper and store embeddings in Chroma."""

    if not state.get("parse_ok"):
        return {
            "embed_ok": False,
            "error": "Cannot embed paper because parsing failed.",
        }

    paper = state["selected_paper"]
    raw_text = state["raw_text"]

    collection_name = collection_name_for(paper.arxiv_id)


    if collection_exists_and_populated(collection_name):
        collection = get_or_create_collection(collection_name)

        return {
            "vectorstore_ref": collection_name,
            "chunk_count": collection.count(),
            "embed_ok": True,
        }

    chunks = chunk_text(raw_text)

    if not chunks:
        return {
            "embed_ok": False,
            "error": "No usable chunks were created from the paper.",
        }

    metadatas = [
        {
            "arxiv_id": paper.arxiv_id,
            "chunk_index": index,
        }
        for index in range(len(chunks))
    ]

    ids = [
        f"{paper.arxiv_id}_chunk_{index}"
        for index in range(len(chunks))
    ]

    try:
        store_chunks(
            collection_name=collection_name,
            chunks=chunks,
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as exc:
        return {
            "embed_ok": False,
            "error": f"Failed to store embeddings: {exc}",
        }

    return {
        "vectorstore_ref": collection_name,
        "chunk_count": len(chunks),
        "embed_ok": True,
    }


def route_embed(state) -> str:
    if state.get("embed_ok"):
        return "summarize"

    return "handle_embedding_failure"


def retrieve(
    vectorstore_ref: str,
    query: str,
    k: int = _DEFAULT_K,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for a query."""

    if not query.strip():
        return []

    results = query_collection(
        vectorstore_ref,
        query,
        k=k,
    )

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    if not documents or not documents[0]:
        return []

    retrieved = []

    for document, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
    ):
        retrieved.append(
            RetrievedChunk(
                text=document,
                metadata=metadata or {},
                distance=float(distance),
            )
        )

    return retrieved


def find_term_in_chunks(
    vectorstore_ref: str,
    term: str,
) -> list[RetrievedChunk]:

    collection = get_or_create_collection(vectorstore_ref)
    results = collection.get(include=["documents", "metadatas"])

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []



    normalized_term = term.lower()
    matches = []

    for document, metadata in zip(documents, metadatas):
        if normalized_term in document.lower():

            matches.append(
                RetrievedChunk(
                    text=document,
                    metadata=metadata or {},
                    distance=0.0,
                )
            )

    return matches


def retrieve_for_qa(
    vectorstore_ref: str,
    question: str,
) -> list[RetrievedChunk]:

    semantic_chunks = retrieve(vectorstore_ref, question, k=_QA_K)

    semantic_candidates = [
        chunk
        for chunk in semantic_chunks
        if chunk["distance"] < _QA_DISTANCE_THRESHOLD
    ]

    lexical_chunks = []
    terms = _extract_lexical_terms(question)

    for term in terms:
        matches = find_term_in_chunks(vectorstore_ref, term)
        lexical_chunks.extend(matches)

    merged = []
    seen_texts = set()


    for chunk in semantic_candidates:
        if chunk["text"] not in seen_texts:
            seen_texts.add(chunk["text"])
            merged.append(chunk)


    for chunk in lexical_chunks:
        if chunk["text"] not in seen_texts:
            seen_texts.add(chunk["text"])
            merged.append(chunk)

    return merged[:8]

def retrieve_for_query(state) -> dict:
    """Run an initial retrieval stage for the graph."""

    paper = state["selected_paper"]

    query = state.get("parsed_query") or state["user_input"]

    chunks = retrieve(
        collection_name_for(paper.arxiv_id),
        query,
    )

    return {
        "retrieved_chunks": chunks,
    }