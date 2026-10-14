
import csv
import json
from pathlib import Path

input_path = Path("qa_evaluation_results.json")
output_path = Path("qa_manual_review.csv")

if not input_path.exists():
    raise FileNotFoundError(
        "qa_evaluation_results.json not found. "
        "Run this script from your project root."
    )

results = json.loads(input_path.read_text(encoding="utf-8"))

columns = [
    "question_id",
    "question",
    "method",
    "expected_chunks",
    "retrieved_chunks",
    "evidence_recall",
    "answer",
    "correctness_0_to_2",
    "citation_0_to_2",
    "grounding_0_to_2",
    "abstention_0_to_2",
    "review_notes",
]

with output_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for result in results:
        writer.writerow({
            "question_id": result.get("question_id", ""),
            "question": result.get("question", ""),
            "method": result.get("method", ""),
            "expected_chunks": ", ".join(
                map(str, result.get("expected_chunks", []))
            ),
            "retrieved_chunks": ", ".join(
                map(str, result.get("retrieved_chunks", []))
            ),
            "evidence_recall": result.get("evidence_recall", ""),
            "answer": result.get("answer", ""),
            "correctness_0_to_2": "",
            "citation_0_to_2": "",
            "grounding_0_to_2": "",
            "abstention_0_to_2": "",
            "review_notes": "",
        })

print(f"Created: {output_path.resolve()}")
print(f"Rows to review: {len(results)}")
