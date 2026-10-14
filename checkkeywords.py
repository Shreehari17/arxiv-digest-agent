
from src.nodes.retrieval import find_term_in_chunks
from src.vectorstore import collection_name_for

coll = collection_name_for("2610.13455v1")

terms = [
    "safety",
    "robustness",
    "security",
    "metrics",
    "evaluation",
    "benchmark",
    "adversarial"
]

for term in terms:
    matches = find_term_in_chunks(coll, term)
    print(f"\n{'='*50}")
    print(f"TERM: {term} | MATCHES: {len(matches)}")

    for chunk in matches[:3]:
        print(
            f"Chunk {chunk['metadata'].get('chunk_index')}: "
            f"{chunk['text'][:400].replace(chr(10), ' ')}"
        )

