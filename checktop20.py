
from src.nodes.retrieval import retrieve
from src.vectorstore import collection_name_for

coll = collection_name_for("2610.13455v1")
q = "What quantitative metrics can be used to evaluate safety and robustness of tool-using agentic LLMs?"

results = retrieve(coll, q, k=20)

for i, ch in enumerate(results):
    print(f"\n#{i+1} | distance={ch['distance']:.3f} | chunk={ch['metadata'].get('chunk_index')}")
    print(ch["text"][:300].replace("\n", " "))
