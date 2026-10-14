
import fitz
import requests
from pathlib import Path

URL = "https://arxiv.org/pdf/2610.13455v1"
PDF_PATH = Path("data/agentic_llms_debug.pdf")

PDF_PATH.parent.mkdir(parents=True, exist_ok=True)

if not PDF_PATH.exists():
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    PDF_PATH.write_bytes(response.content)

doc = fitz.open(PDF_PATH)

terms = [
    "robustness, error handling",
    "safety and security",
    "tool use",
    "safety",
    "robustness",
    "security",
]

print(f"Total PDF pages: {len(doc)}")

for term in terms:
    matches = []

    for page_index in range(len(doc)):
        text = doc[page_index].get_text()
        if term.casefold() in text.casefold():
            matches.append(page_index + 1)

    print(f"\n{term!r}: {len(matches)} matching pages")
    print("First matching PDF pages:", matches[:20])

    # Show snippets from the first few matching pages
    for page_num in matches[:3]:
        text = doc[page_num - 1].get_text()
        pos = text.casefold().find(term.casefold())
        start = max(0, pos - 150)
        end = min(len(text), pos + 350)
        print(f"\n  PDF page {page_num}:")
        print(" ", text[start:end].replace("\n", " "))

doc.close()
