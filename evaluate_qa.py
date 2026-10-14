
import json
from pathlib import Path

from src.llm import call_llm
from src.prompts import QA_SYSTEM_PROMPT, build_qa_prompt
from src.nodes.retrieval import (
    retrieve_for_qa,
    retrieve_for_qa_hybrid,
)

COLLECTION = "paper_2610_13455v1_test245"
TOP_K = 12

TEST_CASES = [
    {
        "id": "Q1",
        "question": "What quantitative metrics does the paper associate with the Fortress AI safety benchmark?",
        "expected_chunks": [206],
    },
    {
        "id": "Q2",
        "question": "What problems can occur when agents use external tools, and how can they respond to those failures?",
        "expected_chunks": [605],
    },
    {
        "id": "Q3",
        "question": "What does the paper describe about jailbreak attacks, malicious agent tasks, and Rainbow Teaming?",
        "expected_chunks": [612, 613],
    },
    {
        "id": "Q4",
        "question": "What are perplexity and bits-per-byte used to measure, and how does the paper distinguish them?",
        "expected_chunks": [200, 201],
    },
    {
        "id": "Q5",
        "question": "Which metrics does the paper mention for evaluating retrieval systems?",
        "expected_chunks": [623],
    },
    {
        "id": "Q6",
        "question": "What is the AgentHarm benchmark, and how many malicious agent tasks does it contain?",
        "expected_chunks": [612],
    },
    {
        "id": "Q7",
        "question": "How does Rainbow Teaming generate adversarial prompts?",
        "expected_chunks": [613],
    },
    {
        "id": "Q8",
        "question": "What is the observe-reflect-retry approach to tool failure?",
        "expected_chunks": [605],
    },
    {
        "id": "Q9",
        "question": "What do ARS and ORS mean in the Fortress benchmark?",
        "expected_chunks": [206],
    },
    {
        "id": "Q10",
        "question": "What security risk can arise from exposing an LLM's system prompt?",
        "expected_chunks": [612],
    },
    {
        "id": "Q11",
        "question": "Which standard metrics does the paper name for evaluating retrieval systems?",
        "expected_chunks": [623],
    },
    {
        "id": "Q12",
        "question": "Where does the paper mention faithfulness scores in retrieval evaluation?",
        "expected_chunks": [623],
    },
    {
        "id": "Q13",
        "question": "Which metric does the paper list for the AIME 2025 benchmark?",
        "expected_chunks": [205, 206],
    },
    {
        "id": "Q14",
        "question": "What method does Rainbow Teaming use to generate diverse adversarial prompts?",
        "expected_chunks": [613],
    },
    {
        "id": "Q15",
        "question": "What can an agent do after a tool call returns an error?",
        "expected_chunks": [605],
    },
]


def run_one(question, retrieval_fn):
    chunks = retrieval_fn(COLLECTION, question)

    context = "\n\n---\n\n".join(
        f"[Paper chunk {chunk['metadata'].get('chunk_index')}]\n"
        f"{chunk['text']}"
        for chunk in chunks
    )

    prompt = build_qa_prompt(
        question=question,
        context=context,
        history=[],
    )

    answer = call_llm(prompt, system=QA_SYSTEM_PROMPT)

    retrieved_ids = [
        chunk["metadata"].get("chunk_index")
        for chunk in chunks
    ]

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_ids,
        "context": context,
    }


def main():
    results = []

    methods = {
        "semantic_only": retrieve_for_qa,
        "hybrid": retrieve_for_qa_hybrid,
    }

    for case in TEST_CASES:
        print(f"\n{'=' * 70}")
        print(f"{case['id']}: {case['question']}")

        for method_name, retrieval_fn in methods.items():
            print(f"\nRunning {method_name}...")

            try:
                output = run_one(case["question"], retrieval_fn)
                retrieved = set(output["retrieved_chunks"])
                expected = set(case["expected_chunks"])

                result = {
                    "question_id": case["id"],
                    "question": case["question"],
                    "method": method_name,
                    "expected_chunks": sorted(expected),
                    "retrieved_chunks": output["retrieved_chunks"],
                    "evidence_recall": (
                        len(expected & retrieved) / len(expected)
                        if expected else None
                    ),
                    "answer": output["answer"],
                    "context": output["context"],
                    "error": None,
                }

                print("Retrieved:", output["retrieved_chunks"])
                print(
                    "Expected evidence recall:",
                    f"{result['evidence_recall']:.0%}",
                )
                print("Answer:", output["answer"])

            except Exception as exc:
                result = {
                    "question_id": case["id"],
                    "question": case["question"],
                    "method": method_name,
                    "error": str(exc),
                }
                print("ERROR:", exc)

            results.append(result)

    output_path = Path("qa_evaluation_results.json")
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
