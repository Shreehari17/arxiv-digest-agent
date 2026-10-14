
import math
import re
from collections import Counter

from src.nodes.retrieval import retrieve, retrieve_for_qa
from src.vectorstore import get_or_create_collection


COLLECTION = "paper_2610_13455v1_test245"
K = 12
CANDIDATE_K = 20
RRF_K = 60

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
        "question": (
            "Which metrics does the paper mention for evaluating "
            "retrieval systems?"
        ),
        "expected": {623},
    },

    {
        "id": "Q6",
        "question": "What is the AgentHarm benchmark, and how many malicious agent tasks does it contain?",
        "expected": {612},
    },
    {
        "id": "Q7",
        "question": "How does Rainbow Teaming generate adversarial prompts?",
        "expected": {613},
    },
    {
        "id": "Q8",
        "question": "What is the observe-reflect-retry approach to tool failure?",
        "expected": {605},
    },
    {
        "id": "Q9",
        "question": "What do ARS and ORS mean in the Fortress benchmark?",
        "expected": {206},
    },
    {
        "id": "Q10",
        "question": "What security risk can arise from exposing an LLM's system prompt?",
        "expected": {612},
    },
    {
        "id": "Q11",
        "question": "Which standard metrics does the paper name for evaluating retrieval systems?",
        "expected": {623},
    },
    {
        "id": "Q12",
        "question": "Where does the paper mention faithfulness scores in retrieval evaluation?",
        "expected": {623},
    },
    {
        "id": "Q13",
        "question": "Which metric does the paper list for the AIME 2025 benchmark?",
        "expected": {205,206},
    },
    {
        "id": "Q14",
        "question": "What method does Rainbow Teaming use to generate diverse adversarial prompts?",
        "expected": {613},
    },
    {
        "id": "Q15",
        "question": "What can an agent do after a tool call returns an error?",
        "expected": {605},
    },
]




def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_search(query, documents, metadatas, k=20):
    """Return BM25-ranked chunks from the paper collection."""
    tokenized_docs = [tokenize(doc) for doc in documents]
    query_terms = set(tokenize(query))

    n_docs = len(tokenized_docs)
    if not n_docs:
        return []

    doc_lengths = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_lengths) / n_docs or 1.0

    document_frequency = Counter()
    term_frequencies = []

    for tokens in tokenized_docs:
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_frequency.update(frequencies.keys())

    k1 = 1.5
    b = 0.75
    scored = []

    for index, frequencies in enumerate(term_frequencies):
        score = 0.0
        doc_length = doc_lengths[index]

        for term in query_terms:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue

            df = document_frequency[term]
            idf = math.log(
                1 + (n_docs - df + 0.5) / (df + 0.5)
            )

            denominator = tf + k1 * (
                1 - b + b * doc_length / avgdl
            )

            score += idf * (
                tf * (k1 + 1) / denominator
            )

        if score > 0:
            scored.append((score, index))

    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    for score, index in scored[:k]:
        results.append({
            "text": documents[index],
            "metadata": metadatas[index] or {},
            "bm25_score": score,
        })

    return results


def rrf_fuse(semantic_results, lexical_results, k=12):
    """Combine ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    chunks_by_id = {}

    for ranked_list in (semantic_results, lexical_results):
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["metadata"].get("chunk_index")
            if chunk_id is None:
                continue

            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                1.0 / (RRF_K + rank)
            )
            chunks_by_id[chunk_id] = chunk

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    return [
        {
            "chunk_id": chunk_id,
            "rrf_score": scores[chunk_id],
            "text": chunks_by_id[chunk_id]["text"],
        }
        for chunk_id in ranked_ids[:k]
    ]


def evaluate(ranked_ids, expected):
    found = expected.intersection(ranked_ids)
    recall = len(found) / len(expected)

    ranks = [
        ranked_ids.index(chunk_id) + 1
        for chunk_id in found
    ]

    first_rank = min(ranks) if ranks else None
    reciprocal_rank = 1 / first_rank if first_rank else 0.0

    return recall, first_rank, reciprocal_rank


def main():
    collection = get_or_create_collection(COLLECTION)
    data = collection.get(include=["documents", "metadatas"])

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    semantic_recalls = []
    qa_recalls = []
    rrf_recalls = []

    semantic_rrs = []
    qa_rrs = []
    rrf_rrs = []

    for item in QUESTIONS:
        query = item["question"]
        expected = item["expected"]

        semantic = retrieve(
            COLLECTION, query, k=CANDIDATE_K
        )
        qa = retrieve_for_qa(COLLECTION, query)
        lexical = bm25_search(
            query, documents, metadatas, k=CANDIDATE_K
        )

        fused = rrf_fuse(
            semantic, lexical, k=K
        )

        semantic_ids = [
            c["metadata"].get("chunk_index")
            for c in semantic[:K]
        ]
        qa_ids = [
            c["metadata"].get("chunk_index")
            for c in qa[:K]
        ]
        rrf_ids = [c["chunk_id"] for c in fused]

        s_recall, s_rank, s_rr = evaluate(
            semantic_ids, expected
        )
        q_recall, q_rank, q_rr = evaluate(
            qa_ids, expected
        )
        r_recall, r_rank, r_rr = evaluate(
            rrf_ids, expected
        )

        semantic_recalls.append(s_recall)
        qa_recalls.append(q_recall)
        rrf_recalls.append(r_recall)

        semantic_rrs.append(s_rr)
        qa_rrs.append(q_rr)
        rrf_rrs.append(r_rr)

        print(f"\n{'=' * 65}")
        print(f"{item['id']}: {query}")
        print(f"Expected: {sorted(expected)}")

        for name, ids, recall, rank, rr in [
            ("SEMANTIC", semantic_ids, s_recall, s_rank, s_rr),
            ("CURRENT QA", qa_ids, q_recall, q_rank, q_rr),
            ("BM25 + RRF", rrf_ids, r_recall, r_rank, r_rr),
        ]:
            print(f"\n{name}")
            print(f"Chunks: {ids}")
            print(f"Recall@{K}: {recall:.2%}")
            print(f"First relevant rank: {rank}")
            print(f"Reciprocal rank: {rr:.4f}")

    count = len(QUESTIONS)

    print(f"\n{'=' * 65}")
    print("MEAN RESULTS")

    for name, recalls, rrs in [
        ("SEMANTIC", semantic_recalls, semantic_rrs),
        ("CURRENT QA", qa_recalls, qa_rrs),
        ("BM25 + RRF", rrf_recalls, rrf_rrs),
    ]:
        print(
            f"{name:12} "
            f"Recall@{K}: {sum(recalls) / count:.2%} | "
            f"MRR@{K}: {sum(rrs) / count:.4f}"
        )


if __name__ == "__main__":
    main()
