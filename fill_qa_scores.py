
import pandas as pd

CSV_PATH = "qa_manual_review.csv"

# Scores in CSV row order:
# correctness, citation, grounding, abstention
scores = [
    (2,2,2,2), (2,2,1,2),  # Q1
    (2,2,2,2), (2,2,1,2),  # Q2
    (2,2,2,2), (2,2,2,2),  # Q3
    (2,2,2,2), (2,2,2,2),  # Q4
    (2,2,2,2), (2,2,2,2),  # Q5
    (2,2,2,2), (2,2,2,2),  # Q6
    (2,2,2,2), (2,2,2,2),  # Q7
    (2,2,2,2), (2,2,2,2),  # Q8
    (0,2,2,2), (2,2,1,2),  # Q9
    (2,2,2,2), (2,2,1,2),  # Q10
    (2,2,2,2), (2,2,2,2),  # Q11
    (2,2,2,2), (2,2,2,2),  # Q12
    (2,2,2,2), (2,2,2,2),  # Q13
    (2,2,2,2), (2,2,2,2),  # Q14
    (2,2,2,2), (2,2,2,2),  # Q15
]

df = pd.read_csv(CSV_PATH)

columns = [
    "correctness_0_to_2",
    "citation_0_to_2",
    "grounding_0_to_2",
    "abstention_0_to_2",
]

if len(df) != len(scores):
    raise ValueError(f"Expected 30 rows, found {len(df)}")

# Ensure the rows are in the expected question/method order
for i, row in df.iterrows():
    expected_q = f"Q{(i // 2) + 1}"
    expected_method = "semantic_only" if i % 2 == 0 else "hybrid"

    if row["question_id"] != expected_q or row["method"] != expected_method:
        raise ValueError(f"Unexpected row order at row {i + 1}")

df[columns] = scores
df.to_csv(CSV_PATH, index=False)

print(f"Updated {CSV_PATH} with scores for {len(df)} rows.")
print(df.groupby("method")[columns].mean().round(3))

