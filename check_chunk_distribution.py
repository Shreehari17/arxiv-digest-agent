
from src.vectorstore import (
    collection_name_for,
    _get_client,
    _get_embedding_fn,
)

name = collection_name_for("2610.13455v1")
client = _get_client()
collection = client.get_collection(
    name=name,
    embedding_function=_get_embedding_fn(),
)

data = collection.get(include=["documents", "metadatas"])
docs = data["documents"]
metas = data["metadatas"]

if docs and isinstance(docs[0], list):
    docs = docs[0]
if metas and isinstance(metas[0], list):
    metas = metas[0]

# Inspect chunks distributed across the stored document
for i in [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 150]:
    if i < len(docs):
        print(f"\n{'='*60}")
        print(f"CHUNK {metas[i].get('chunk_index')}")
        print(docs[i][:600].replace("\n", " "))
