
from src.nodes.retrieval import retrieve, retrieve_for_qa

COLLECTION = "paper_2610_13455v1_test245"
K = 12

QUESTIONS = [
    {
        "id": "Q1",
        "question": (
            "What quantitative metrics does the paper associate "
            "with the Fortress AI safety benchmark?"
        ),
        "expected": {206},
    },
    {
        "id": "Q2",
        "question": (
            "What problems can occur when agents use external tools, "
            "and how can they respond to those failures?"
        ),
        "expected": {605},
    },
    {
        "id": "Q3",
        "question": (
            "What does the paper describe about jailbreak attacks, "
            "malicious agent tasks, and Rainbow Teaming?"
        ),
        "expected": {612, 613},
    },
    {
        "id": "Q4",
        "question": (
            "What are perplexity and bits-per-byte used to measure, "
            "and how does the paper distinguish them?"
        ),
        "expected": {200, 201},
    },
    {
        "id": "Q5",
        "question":(
            "Which metrics does the paper mention for evaluating "
            "retrieval systems?"
        ),
        "expected": {623},
    },
]


def evaluate(results, expected):
    ranked_ids = [
        chunk["metadata"].get("chunk_index")
        for chunk in results
    ]

    found = expected.intersection(ranked_ids)
    recall = len(found) / len(expected)

    ranks = [
        ranked_ids.index(chunk_id) + 1
        for chunk_id in found
    ]

    first_rank = min(ranks) if ranks else None
    reciprocal_rank = 1 / first_rank if first_rank else 0.0

    return ranked_ids, recall, first_rank, reciprocal_rank


def main():
    semantic_recall = []
    semantic_rr = []
    qa_recall = []
    qa_rr = []

    for item in QUESTIONS:
        semantic = retrieve(COLLECTION, item["question"], k=K)
        qa = retrieve_for_qa(COLLECTION, item["question"])

        s_ids, s_recall, s_rank, s_rr = evaluate(
            semantic, item["expected"]
        )
        q_ids, q_recall, q_rank, q_rr = evaluate(
            qa, item["expected"]
        )

        semantic_recall.append(s_recall)
        semantic_rr.append(s_rr)
        qa_recall.append(q_recall)
        qa_rr.append(q_rr)

        print(f"\n{'=' * 65}")
        print(f"{item['id']}: {item['question']}")
        print(f"Expected chunks: {sorted(item['expected'])}")

        print("\nSEMANTIC ONLY")
        print(f"Retrieved: {s_ids}")
        print(f"Recall@{K}: {s_recall:.2%}")
        print(f"First relevant rank: {s_rank}")
        print(f"Reciprocal rank: {s_rr:.4f}")

        print("\nCURRENT QA RETRIEVAL")
        print(f"Retrieved: {q_ids}")
        print(f"Recall@{K}: {q_recall:.2%}")
        print(f"First relevant rank: {q_rank}")
        print(f"Reciprocal rank: {q_rr:.4f}")

    count = len(QUESTIONS)

    print(f"\n{'=' * 65}")
    print("MEAN COMPARISON")
    print(f"Semantic Recall@{K}: {sum(semantic_recall) / count:.2%}")
    print(f"QA Recall@{K}:       {sum(qa_recall) / count:.2%}")
    print(f"Semantic MRR@{K}:    {sum(semantic_rr) / count:.4f}")
    print(f"QA MRR@{K}:          {sum(qa_rr) / count:.4f}")


if __name__ == "__main__":
    main()
