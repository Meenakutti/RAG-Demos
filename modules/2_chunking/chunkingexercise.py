import json
import os
import sys
import warnings
# ensure we aren't running under Python 3.14+ because chromadb uses pydantic v1
# which currently explodes with an inference error (see https://github.com/chroma-core/chroma/issues/…)
if sys.version_info >= (3, 14):
    raise RuntimeError(
        "Chroma vector store is incompatible with Python 3.14 or later.\n"
        "Please use Python 3.13 or earlier, or downgrade your interpreter." 
    )

# suppress known pydantic warning on Python 3.14+
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from pathlib import Path
DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "synthetic_tickets.json"

load_dotenv()

# Load data
with DATA_FILE.open("r", encoding="utf-8") as f:
    tickets = json.load(f)

documents = [
    Document(
        page_content=f"{t['title']}. {t['description']}",
        metadata={'ticket_id': t['ticket_id'], 'category': t['category']}
    )
    for t in tickets
]

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# Step 1: Build and save (Chroma persists with persist_directory)
print("Building vector store...")
store = Chroma.from_documents(
    documents, 
    embeddings, 
    collection_name="my_collection",
    persist_directory="./my_chroma_db"
)
print("✓ Saved to ./my_chroma_db")

# Step 2: Load it back
print("\nLoading vector store...")
loaded_store = Chroma(
    persist_directory="./my_chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection"
)
print("✓ Loaded from disk")

# Step 3: Verify it works
query = "login problem"
results = loaded_store.similarity_search(query, k=3)
print(f"\nSearch results for '{query}':")
for doc in results:
    print(f"  {doc.metadata['ticket_id']}: {doc.page_content[:50]}...")
