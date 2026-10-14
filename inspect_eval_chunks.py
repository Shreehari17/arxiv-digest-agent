
from src.vectorstore import get_or_create_collection

COLLECTION = "paper_2610_13455v1_test245"

CHUNK_IDS_TO_CHECK = [204, 205, 206]

collection = get_or_create_collection(COLLECTION)

results = collection.get(
    where={"chunk_index": {"$in": CHUNK_IDS_TO_CHECK}},
    include=["documents", "metadatas"],
)

for metadata, document in zip(
    results["metadatas"],
    results["documents"],
):
    chunk_id = metadata["chunk_index"]

    print("\n" + "=" * 80)
    print(f"CHUNK {chunk_id}")
    print("=" * 80)
    print(document[:1800])

