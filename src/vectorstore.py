import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


logger = logging.getLogger(__name__)

_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"

_EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

_client = None
_embedding_fn = None


def _get_client():
    global _client

    if _client is None:
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(_PERSIST_DIR)
        )

    return _client


def _get_embedding_fn():
    global _embedding_fn

    if _embedding_fn is None:
        _embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=_EMBEDDING_MODEL_NAME
        )

    return _embedding_fn


def collection_name_for(arxiv_id: str) -> str:
    """Create a safe, deterministic Chroma collection name."""

    safe_id = arxiv_id.replace(".", "_").replace("/", "_")

    return f"paper_{safe_id}"


def get_or_create_collection(name: str):
    client = _get_client()

    return client.get_or_create_collection(
        name=name,
        embedding_function=_get_embedding_fn(),
    )


def collection_exists_and_populated(name: str) -> bool:
    """Check whether embeddings for this paper already exist."""

    client = _get_client()

    try:
        collection = client.get_collection(
            name=name,
            embedding_function=_get_embedding_fn(),
        )
    except Exception:
        return False

    return collection.count() > 0


def store_chunks(
    collection_name: str,
    chunks: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """Store chunks in Chroma. Chroma generates their embeddings."""

    collection = get_or_create_collection(collection_name)

    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )


def query_collection(
    collection_name: str,
    query_text: str,
    k: int,
) -> dict:
    """Find the k chunks most semantically similar to a query."""

    collection = get_or_create_collection(collection_name)

    return collection.query(
        query_texts=[query_text],
        n_results=k,
    )