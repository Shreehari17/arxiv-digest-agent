from src.nodes.retrieval import retrieve_for_qa_hybrid

COLLECTION = "paper_2610_13455v1_test245"

questions = [
    "What do ARS and ORS mean in the Fortress benchmark?",
    "Which standard metrics does the paper name for evaluating retrieval systems?",
    "What can an agent do after a tool call returns an error?",
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    chunks = retrieve_for_qa_hybrid(
        COLLECTION,
        question,
        k=12,
    )

    for rank, chunk in enumerate(chunks, start=1):
        print(
            f"\nRank {rank} | "
            f"Chunk {chunk['metadata'].get('chunk_index')} | "
            f"Distance {chunk['distance']:.4f}"
        )
        print(chunk["text"][:250].replace("\n", " "))
