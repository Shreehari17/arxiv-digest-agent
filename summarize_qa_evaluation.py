
import pandas as pd

df = pd.read_csv("qa_manual_review.csv")

criteria = [
    "correctness_0_to_2",
    "citation_0_to_2",
    "grounding_0_to_2",
    "abstention_0_to_2",
]

# Make sure all scores have been filled
if df[criteria].isna().any().any():
    print("Some scores are missing. Fill all four scoring columns first.")
    print(df[df[criteria].isna().any(axis=1)][
        ["question_id", "method"] + criteria
    ].to_string(index=False))
    raise SystemExit(1)

summary = df.groupby("method")[criteria].mean().round(3)
summary.columns = [
    "Correctness",
    "Citation accuracy",
    "Grounding",
    "Appropriate abstention",
]

summary["Overall average (/2)"] = summary.mean(axis=1)

print("\n=== QA EVALUATION SUMMARY ===")
print(summary.to_string())

print("\n=== HYBRID IMPROVEMENT OVER SEMANTIC-ONLY ===")
for metric in summary.columns:
    difference = summary.loc["hybrid", metric] - summary.loc[
        "semantic_only", metric
    ]
    print(f"{metric}: {difference:+.3f}")
