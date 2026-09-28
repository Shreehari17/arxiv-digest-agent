from src.llm import call_llm
from src.prompts import (
    QA_SYSTEM_PROMPT,
    build_qa_prompt,
)
from src.state import AgentState, QATurn
from src.nodes.retrieval import (
    retrieve_for_qa,
)


def answer_question(
    question: str,
    vectorstore_ref: str,
    history: list[QATurn],
) -> QATurn:
    """
    Answer one question using retrieved paper chunks.
    """


    retrieval_question = question

    if history:
        previous_turn = history[-1]

        retrieval_question = (
            f"Previous question: {previous_turn.question}\n"
            f"Previous answer: {previous_turn.answer}\n"
            f"Current question: {question}"
        )

    chunks = retrieve_for_qa(
        vectorstore_ref,
        retrieval_question,
    )


    if not chunks:
        return QATurn(
            question=question,
            answer=(
                "I couldn't find content in this paper "
                "related to that question."
            ),
        )

    context_parts = []

    for chunk in chunks:
        chunk_index = chunk["metadata"].get("chunk_index")

        context_parts.append(
            f"[Paper chunk {chunk_index}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = build_qa_prompt(
        question=question,
        context=context,
        history=history,
    )

    answer = call_llm(
        prompt,
        system=QA_SYSTEM_PROMPT,
    )

    return QATurn(
        question=question,
        answer=answer,
    )


def qa_loop(state: AgentState) -> dict:
    """
    Ask the user questions about the current paper.

    Commands:
        /paper <arXiv ID or topic>  -> switch to another paper
        /quit                       -> exit the agent
    """


    if not state.get("conversation_history"):

        briefing = state.get("briefing")

        if briefing:
            print("\n" + "=" * 60)
            print("PAPER BRIEFING")
            print("=" * 60)

            print(f"\nTitle: {briefing.title}")
            print(f"Authors: {', '.join(briefing.authors)}")
            print(f"arXiv ID: {briefing.arxiv_id}")
            print(f"Published: {briefing.published}")
            print(f"PDF: {briefing.pdf_url}")
            if state.get("truncated", False):
                print(
                "\n⚠️ Note: This paper is longer than the configured "
                "page limit. Only the first 60 pages were parsed, "
                "so answers are limited to the parsed portion."
                )

            print("\nWhy it matters:")
            print(briefing.why_it_matters)

            print("\nProblem statement:")
            print(briefing.problem_statement)

            print("\nMethods:")
            for method in briefing.methods:
                print(f"- {method}")

            print("\nKey results:")
            for result in briefing.key_results:
                print(f"- {result}")

            print("\nLimitations:")
            for limitation in briefing.limitations:
                print(f"- {limitation}")

            print("\nSuggested follow-up questions:")
            for question in briefing.follow_up_questions:
                print(f"- {question}")

    print("\n" + "=" * 60)
    print("                       YOUR QUESTION")
    print("=" * 60)
    user_input = input("❯ ").strip()


    if not user_input or user_input.lower() in {"/quit", "quit", "exit"}:
        return {
        "continue_qa": False,
        "next_action": "end",
    }


    if user_input.lower().startswith("/paper"):
        new_input = user_input[len("/paper"):].strip()

        if not new_input:
            print(
                "\nUsage: /paper <arXiv ID or research topic>"
            )
            return {
                "continue_qa": True,
            }

        print(
            f"\nSwitching paper to: {new_input}"
        )

        return {
        "user_input": new_input,
        "intent": None,
        "parsed_query": None,
        "briefing": None,
        "candidates": [],
        "selected_paper": None,
        "raw_text": None,
        "parse_ok": False,
        "truncated": False,
        "vectorstore_ref": None,
        "chunk_count": 0,
        "embed_ok": False,
        "retrieved_chunks": [],
        "conversation_history": [],
        "continue_qa": False,
        "last_answer": None,
        "error": None,
        "next_action": "change_paper",
    }

    history = state.get(
        "conversation_history",
        []
    )

    vectorstore_ref = state["vectorstore_ref"]

    turn = answer_question(
        question=user_input,
        vectorstore_ref=vectorstore_ref,
        history=history,
    )

    updated_history = history + [turn]

    print("\nAnswer:")
    print(turn.answer)

    return {
    "conversation_history": updated_history,
    "continue_qa": True,
    "last_answer": turn.answer,
    "next_action": "qa",
    }


def route_qa(state: AgentState) -> str:
    action = state.get("next_action")

    if action == "change_paper":
        return "change_paper"

    if action == "qa":
        return "qa_loop"

    return "end"