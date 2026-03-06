from pathlib import Path
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# load tickets
DATA_FILE = Path('modules/4_rag_pipeline/demo.py').resolve().parents[2]/'data'/'synthetic_tickets.json'
with DATA_FILE.open('r', encoding='utf-8') as f:
    tickets = json.load(f)

embeddings=OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = Chroma.from_documents(
    documents=[Document(page_content=f"{t['ticket_id']}", metadata={'ticket_id':t['ticket_id']}) for t in tickets],
    embedding=embeddings,
    collection_name="supportdesk_rag",
    persist_directory="./rag_vectorstore"
)
print('count', vector_store._collection.count())
