
import ast
import re
import pandas as pd

CSV_PATH = "qa_manual_review.csv"
K = 12

df = pd.read_csv(CSV_PATH)


def parse_chunk_ids(value):
    """Extract chunk IDs from lists, sets, JSON-like strings, or comma-separated IDs."""
    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [int(x) for x in value]

    text = str(value).strip()

    # First try parsing structured representations
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [int(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass

    # Handles strings such as "{206, 605}" or "206, 605"
    return [int(x) for x in re.findall(r"\d+", text)]


results = []

for _, row in df.iterrows():
    expected = set(parse_chunk_ids(row["expected_chunks"]))
    retrieved = parse_chunk_ids(row["retrieved_chunks"])[:K]

    # Avoid counting duplicate retrieved IDs more than once
    retrieved_unique = list(dict.fromkeys(retrieved))
    relevant_found = set(retrieved_unique) & expected

    precision = (
        len(relevant_found) / len(retrieved_unique)
        if retrieved_unique else 0.0
    )

    recall = (
        len(relevant_found) / len(expected)
        if expected else float("nan")
    )

    results.append({
        "question_id": row["question_id"],
        "method": row["method"],
        "relevant_found": len(relevant_found),
        "expected_relevant": len(expected),
        "retrieved_count": len(retrieved_unique),
        "chunk_precision_at_12": precision,
        "chunk_recall_at_12": recall,
    })

result_df = pd.DataFrame(results)

print("\n=== PER-QUESTION CONTEXT METRICS ===")
print(result_df.to_string(index=False))

summary = result_df.groupby("method")[
    ["chunk_precision_at_12", "chunk_recall_at_12"]
].mean().round(4)

print("\n=== MACRO-AVERAGE CONTEXT METRICS ===")
print(summary)

result_df.to_csv("context_metrics_per_question.csv", index=False)
summary.to_csv("context_metrics_summary.csv")

print("\nSaved context_metrics_per_question.csv")
print("Saved context_metrics_summary.csv")
