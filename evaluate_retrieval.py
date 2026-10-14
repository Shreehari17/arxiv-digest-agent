
from src.nodes.retrieval import retrieve

COLLECTION = "paper_2610_13455v1_test245"
K = 12

EVALUATION_QUESTIONS = [
    {
        "id": "Q1",
        "question": (
            "What quantitative metrics does the paper associate "
            "with the Fortress AI safety benchmark?"
        ),
        "expected_chunks": {206},
    },
    {
        "id": "Q2",
        "question": (
            "What problems can occur when agents use external tools, "
            "and how can they respond to those failures?"
        ),
        "expected_chunks": {605},
    },
    {
        "id": "Q3",
        "question": (
            "What does the paper describe about jailbreak attacks, "
            "malicious agent tasks, and Rainbow Teaming?"
        ),
        "expected_chunks": {612, 613},
    },
    {
        "id": "Q4",
        "question": (
            "What are perplexity and bits-per-byte used to measure, "
            "and how does the paper distinguish them?"
        ),
        "expected_chunks": {200, 201},
    },
    {
        "id": "Q5",
        "question": (
            "Which metrics does the paper mention for evaluating "
            "retrieval systems?"
        ),
        "expected_chunks": {623},
    },
]


def main():
    total_recall = 0.0
    total_reciprocal_rank = 0.0

    print(f"Collection: {COLLECTION}")
    print(f"Retrieval depth: top {K}")
    print("=" * 70)

    for item in EVALUATION_QUESTIONS:
        results = retrieve(
            COLLECTION,
            item["question"],
            k=K,
        )

        ranked_chunks = [
            chunk["metadata"].get("chunk_index")
            for chunk in results
        ]

        expected = item["expected_chunks"]
        found = expected.intersection(ranked_chunks)

        recall = len(found) / len(expected)
        total_recall += recall

        relevant_ranks = [
            ranked_chunks.index(chunk_id) + 1
            for chunk_id in found
        ]

        first_relevant_rank = (
            min(relevant_ranks) if relevant_ranks else None
        )

        reciprocal_rank = (
            1 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        total_reciprocal_rank += reciprocal_rank

        print(f"\n{item['id']}: {item['question']}")
        print(f"Expected chunks: {sorted(expected)}")
        print(f"Retrieved chunks: {ranked_chunks}")
        print(f"Relevant chunks found: {sorted(found)}")
        print(f"Relevant chunk ranks: {sorted(relevant_ranks)}")
        print(f"Recall@{K}: {recall:.2%}")
        print(f"First relevant rank: {first_relevant_rank}")
        print(f"Reciprocal rank: {reciprocal_rank:.4f}")

    count = len(EVALUATION_QUESTIONS)

    print("\n" + "=" * 70)
    print(f"Mean Recall@{K}: {total_recall / count:.2%}")
    print(f"MRR@{K}: {total_reciprocal_rank / count:.4f}")


if __name__ == "__main__":
    main()
