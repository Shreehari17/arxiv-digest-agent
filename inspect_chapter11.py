
from src.vectorstore import (
    _get_client,
    _get_embedding_fn,
)

COLLECTION = "paper_2610_13455v1_test245"

client = _get_client()
collection = client.get_collection(
    name=COLLECTION,
    embedding_function=_get_embedding_fn(),
)

data = collection.get(
    include=["documents", "metadatas"]
)

docs = data["documents"] or []
metas = data["metadatas"] or []

terms = [
    "success rate",
    "failure rate",
    "attack success rate",
    "robustness",
    "safety",
    "tool error",
    "recovery",
    "metric",
    "benchmark",
]

for term in terms:
    print(f"\n{'=' * 65}")
    print(f"TERM: {term}")

    matches = []
    for doc, meta in zip(docs, metas):
        if term.casefold() in doc.casefold():
            matches.append((doc, meta))

    print(f"Matches: {len(matches)}")

    for doc, meta in matches[:4]:
        print(f"\nChunk {meta.get('chunk_index')}")
        print(doc[:900].replace("\n", " "))
