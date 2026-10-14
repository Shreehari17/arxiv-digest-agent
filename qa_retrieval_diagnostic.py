# qa_retrieval_diagnostic.py  -- run in repo root: python qa_retrieval_diagnostic.py
from pprint import pprint
from src.nodes.retrieval import (
    retrieve,
    retrieve_for_qa,
    find_term_in_chunks,
    _extract_lexical_terms,
    _QA_K,
    _QA_DISTANCE_THRESHOLD,
)
from src.vectorstore import collection_name_for

# Put one of your actual failing follow-up questions here:
QUESTION = (
    "What quantitative metrics can be used to evaluate "
    "safety and robustness of tool-using agentic LLMs?"
)

COLLECTION = collection_name_for("2610.13455v1")  # use v1 collection since it is populated

print("Collection:", COLLECTION)
print("Question:", QUESTION)
print("QA k (semantic):", _QA_K)
print("QA distance threshold:", _QA_DISTANCE_THRESHOLD)
print("\n--- Semantic retrieval (retrieve with k=_QA_K) ---")
sem_results = retrieve(COLLECTION, QUESTION, k=_QA_K)
print("retrieve() returned", len(sem_results), "chunks")
for i, ch in enumerate(sem_results):
    print(f"\n[semantic #{i}] distance={ch['distance']}, metadata={ch.get('metadata')}")
    preview = ch['text'][:500].replace("\n", " ")
    print(" preview:", preview)

print("\n--- Which semantic results survive the QA distance filter (< threshold) ---")
survivors = [ch for ch in sem_results if ch['distance'] < _QA_DISTANCE_THRESHOLD]
print("survivors count:", len(survivors))
for i, ch in enumerate(survivors):
    print(f"[survivor #{i}] distance={ch['distance']}, chunk_index={ch['metadata'].get('chunk_index')}")
    print(" preview:", ch['text'][:300].replace("\n", " "))

print("\n--- Lexical extraction and matches ---")
terms = _extract_lexical_terms(QUESTION)
print("extracted lexical terms:", terms)
lex_matches = []
for term in terms:
    matches = find_term_in_chunks(COLLECTION, term)
    print(f"term '{term}' -> {len(matches)} matches")
    for m in matches[:3]:
        print("  metadata:", m['metadata'], "distance:", m['distance'])
        print("  preview:", m['text'][:200].replace('\n',' '))
    lex_matches.extend(matches)

print("\n--- Merging and deduping behavior (retrieve_for_qa) ---")
merged = retrieve_for_qa(COLLECTION, QUESTION)
print("retrieve_for_qa merged and returned:", len(merged), "chunks (limit 8)")
for i, ch in enumerate(merged):
    print(f"[merged #{i}] distance={ch['distance']}, metadata={ch.get('metadata')}")
    print("  preview:", ch['text'][:400].replace('\\n',' '))

print("\n--- Reconstructed QA prompt (what will be passed to the LLM) ---")
from src.prompts import build_qa_prompt
from src.state import QATurn
from src.nodes.qa import answer_question
# Build context exactly as answer_question would:
context_parts = []
for chunk in merged:
    chunk_index = chunk["metadata"].get("chunk_index")
    context_parts.append(f"[Paper chunk {chunk_index}]\n{chunk['text']}")
context = "\n\n---\n\n".join(context_parts)
prompt = build_qa_prompt(question=QUESTION, context=context, history=[])
print("Prompt preview (first 2000 chars):\n")
print(prompt[:2000].replace("\n", " "))
print("\n\nEnd of prompt preview.")