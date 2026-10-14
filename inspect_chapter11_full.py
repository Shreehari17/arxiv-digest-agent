
from src.vectorstore import _get_client, _get_embedding_fn

COLLECTION = "paper_2610_13455v1_test245"

client = _get_client()
collection = client.get_collection(
    name=COLLECTION,
    embedding_function=_get_embedding_fn(),
)

data = collection.get(include=["documents", "metadatas"])

items = sorted(
    zip(data["documents"] or [], data["metadatas"] or []),
    key=lambda item: item[1].get("chunk_index", -1),
)

for doc, meta in items:
    idx = meta.get("chunk_index", -1)

    # Inspect chunks around the relevant Chapter 11 section.
    if 611 <= idx <= 618:
        print(f"\n{'=' * 70}")
        print(f"CHUNK {idx}")
        print(doc)
