
from src.nodes.retrieval import retrieve, retrieve_for_qa
from src.nodes.retrieval import find_term_in_chunks
COLLECTION = "paper_2610_13455v1_test245"

question = (
    "AgentHarm benchmark, malicious agent tasks, "
    "jailbreak attacks, adversarial prompts, "
    "Rainbow Teaming, safety and security "
    "of agentic LLMs"
)

print("\n=== DIRECT SEMANTIC RETRIEVAL ===")

direct = retrieve(COLLECTION, question, k=15)

for rank, chunk in enumerate(direct, start=1):
    print(
        f"Rank {rank} | "
        f"Chunk {chunk['metadata'].get('chunk_index')} | "
        f"Distance {chunk['distance']:.4f}"
    )

print("\n=== ACTUAL QA RETRIEVAL ===")

qa_chunks = retrieve_for_qa(COLLECTION, question)

for rank, chunk in enumerate(qa_chunks, start=1):
    print(
        f"Rank {rank} | "
        f"Chunk {chunk['metadata'].get('chunk_index')} | "
        f"Distance {chunk['distance']:.4f}"
    )
print("\n=== LEXICAL SEARCH TEST ===")

for term in ["AgentHarm", "Rainbow Teaming", "jailbreak"]:
    matches = find_term_in_chunks(COLLECTION, term)

    print(f"\nTerm: {term}")
    print(f"Matches: {len(matches)}")

    for chunk in matches[:5]:
        print(
            f"Chunk {chunk['metadata'].get('chunk_index')} | "
            f"{chunk['text'][:120]!r}"
        )

from src.vectorstore import get_or_create_collection

collection = get_or_create_collection(COLLECTION)
data = collection.get(include=["documents", "metadatas"])

print("\n=== VERIFY ANSWER SOURCES ===")

for document, metadata in zip(
    data["documents"],
    data["metadatas"],
):
    chunk_id = metadata.get("chunk_index")

    if chunk_id in [200, 201, 206, 623]:
        print(f"\n{'=' * 20} CHUNK {chunk_id} {'=' * 20}")
        print(document)
