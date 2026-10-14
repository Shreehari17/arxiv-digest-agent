
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

load_dotenv()

INPUT_FILE = Path("qa_evaluation_results.json")
OUTPUT_FILE = Path("ragas_evaluation_results.csv")

MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "all-mpnet-base-v2"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.resolve()}. "
            "Place qa_evaluation_results.json in the project root."
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from your .env file.")

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    print(f"Loaded {len(records)} QA records.")

    # Connect Ragas to the existing Groq model.
    groq_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

   
    judge_llm = llm_factory(
        model=MODEL,
        provider="openai",
        client=groq_client,
        temperature=0,
        max_tokens=4096,
        )

    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL,
        device="cpu",
        normalize_embeddings=True,
    )

    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(
        llm=judge_llm,
        embeddings=embeddings,
    )

    results = []

    for index, record in enumerate(records, start=1):
        question = record.get("question", "")
        answer = record.get("answer", "")
        context_text = record.get("context", "")

        # Skip records that failed during the original evaluation.
        if record.get("error") or not question or not answer or not context_text:
            results.append({
                "question_id": record.get("question_id"),
                "method": record.get("method"),
                "faithfulness": None,
                "answer_relevancy": None,
                "error": record.get("error") or "Missing question, answer, or context",
            })
            continue

        # The saved context is the exact context used to generate the answer.
        # Split it back into individual paper chunks.
        contexts = [
            part.strip()
            for part in context_text.split("\n\n---\n\n")
            if part.strip()
        ]

        sample = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }

        print(
            f"[{index}/{len(records)}] "
            f"{record.get('question_id')} | {record.get('method')}"
        )

        faithfulness_score = None
        relevancy_score = None
        errors = []

        try:
            faithfulness_result = faithfulness.score(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            )
            faithfulness_score=faithfulness_result.value
        except Exception as exc:
            errors.append(f"Faithfulness: {exc}")
            print(f"  Faithfulness failed: {exc}")

        try:
            relevancy_result = answer_relevancy.score(
            user_input=question,
            response=answer,
            )
            relevancy_score = relevancy_result.value
        except Exception as exc:
            errors.append(f"Answer relevancy: {exc}")
            print(f"  Answer relevancy failed: {exc}")

        results.append({
            "question_id": record.get("question_id"),
            "method": record.get("method"),
            "faithfulness": (
                float(faithfulness_score)
                if faithfulness_score is not None else None
            ),
            "answer_relevancy": (
                float(relevancy_score)
                if relevancy_score is not None else None
            ),
            "error": "; ".join(errors),
        })
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    # Save results separately; the original JSON remains untouched.
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved scores to {OUTPUT_FILE.resolve()}")

    print("\nMean scores by retrieval method:")
    print(
        df.groupby("method")[
            ["faithfulness", "answer_relevancy"]
        ].mean().round(4)
    )


if __name__ == "__main__":
    main()
