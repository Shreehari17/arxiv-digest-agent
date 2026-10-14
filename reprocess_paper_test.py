
import os
import sys
import requests
import pymupdf

from src.nodes.retrieval import chunk_text
from src.vectorstore import (
    _get_client,
    _get_embedding_fn,
    get_or_create_collection,
    store_chunks,
    collection_name_for,
)

ARXIV_ID = "2610.13455v1"
PDF_URL = "https://arxiv.org/pdf/2610.13455v1"
TEST_COLLECTION = collection_name_for(ARXIV_ID) + "_test245"
PAGE_LIMIT = 245

# Safety check: abort if we cannot verify the collection list.
try:
    client = _get_client()
    existing = client.list_collections()
    names = {
        c["name"] if isinstance(c, dict) else c.name
        for c in existing
    }
except Exception as exc:
    print("Could not verify existing collections; aborting:", exc)
    sys.exit(1)

if TEST_COLLECTION in names:
    print(f"Collection {TEST_COLLECTION!r} already exists.")
    print("Aborting to protect existing data. Do not delete it automatically.")
    sys.exit(1)

# Download and extract the first 120 pages.
print("Downloading PDF...")
response = requests.get(PDF_URL, timeout=60)
response.raise_for_status()

doc = pymupdf.open(stream=response.content, filetype="pdf")
print("PDF pages:", doc.page_count)

pages_to_read = min(PAGE_LIMIT, doc.page_count)
print("Pages to parse:", pages_to_read)

text_parts = [
    doc.load_page(i).get_text()
    for i in range(pages_to_read)
]
doc.close()

raw_text = "\n".join(text_parts)

if len(raw_text.strip()) < 500:
    print("Too little text extracted; aborting.")
    sys.exit(1)

chunks = chunk_text(raw_text)
print("Chunks created:", len(chunks))

if not chunks:
    print("No chunks created; aborting.")
    sys.exit(1)

metadatas = [
    {"arxiv_id": ARXIV_ID, "chunk_index": i}
    for i in range(len(chunks))
]
ids = [f"{ARXIV_ID}_test245_chunk_{i}" for i in range(len(chunks))]

# Create a separate collection and store the new chunks.
collection = get_or_create_collection(TEST_COLLECTION)
store_chunks(TEST_COLLECTION, chunks, metadatas, ids)

print("\nReprocessing complete.")
print("Test collection:", TEST_COLLECTION)
print("Stored chunks:", collection.count())
print("Original collection was not intentionally modified.")

