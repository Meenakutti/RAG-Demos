# -*- coding: utf-8 -*-
"""
Indexing Strategies Demo
========================

This demo teaches:
1. Different FAISS index types and their trade-offs
2. Exact vs approximate search
3. Index optimization techniques
4. Comparing performance across index types
"""

import json
import time
import numpy as np
import os
from openai import OpenAI
import faiss
from langchain_community.vectorstores import Chroma, FAISS as LangChainFAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*80)
print("INDEXING STRATEGIES")
print("="*80)

# Load tickets
with open('../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"\nLoaded {len(tickets)} support tickets")

# Create documents
documents = []
for ticket in tickets:
    full_text = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()
    
    doc = Document(
        page_content=full_text,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )
    documents.append(doc)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
embedding_dim = 1536  # text-embedding-3-small dimension
texts = [doc.page_content for doc in documents]
print(f"\nEncoding {len(texts)} documents...")
response = client.embeddings.create(input=texts, model=embedding_model)
embeddings = np.array([data.embedding for data in response.data])
embeddings_np = embeddings.astype('float32')

# ============================================================================
# PART 1: Exact Search Indexes
# ============================================================================
print("\n" + "="*80)
print("PART 1: Exact Search Indexes")
print("="*80)

# Index Type 1: Flat L2 (Euclidean distance)
print("\n--- Index 1: IndexFlatL2 (Exact L2 Distance) ---")
index_flat_l2 = faiss.IndexFlatL2(embedding_dim)
index_flat_l2.add(embeddings_np)
print(f"✓ Created IndexFlatL2 with {index_flat_l2.ntotal} vectors")
print(f"  - Always finds exact nearest neighbors")
print(f"  - Uses L2 (Euclidean) distance")
print(f"  - Good for: Small to medium datasets (<100K vectors)")

# Index Type 2: Flat IP (Inner Product / Cosine Similarity)
print("\n--- Index 2: IndexFlatIP (Exact Inner Product) ---")
# Normalize embeddings for cosine similarity
embeddings_normalized = embeddings_np.copy()
faiss.normalize_L2(embeddings_normalized)
index_flat_ip = faiss.IndexFlatIP(embedding_dim)
index_flat_ip.add(embeddings_normalized)
print(f"✓ Created IndexFlatIP with {index_flat_ip.ntotal} vectors")
print(f"  - Uses inner product (cosine similarity when normalized)")
print(f"  - Better for semantic similarity")
print(f"  - Normalized embeddings recommended")

# Compare results
query = "Authentication problems after password reset"
query_response = client.embeddings.create(input=[query], model=embedding_model)
query_embedding = np.array([query_response.data[0].embedding], dtype='float32')
query_normalized = query_embedding.copy()
faiss.normalize_L2(query_normalized)

print(f"\nTest query: '{query}'")

# Search with L2
D_l2, I_l2 = index_flat_l2.search(query_embedding, 3)
print("\nL2 Results:")
for i, (idx, dist) in enumerate(zip(I_l2[0], D_l2[0]), 1):
    print(f"  #{i} - Distance: {dist:.4f} - {tickets[idx]['ticket_id']}")

# Search with IP
D_ip, I_ip = index_flat_ip.search(query_normalized, 3)
print("\nInner Product Results:")
for i, (idx, score) in enumerate(zip(I_ip[0], D_ip[0]), 1):
    print(f"  #{i} - Score: {score:.4f} - {tickets[idx]['ticket_id']}")

# ============================================================================
# PART 2: Approximate Search Indexes (for Scale)
# ============================================================================
print("\n" + "="*80)
print("PART 2: Approximate Search Indexes")
print("="*80)

# Index Type 3: IVF (Inverted File Index)
print("\n--- Index 3: IndexIVFFlat (Inverted File) ---")
nlist = 10  # Number of clusters (use sqrt(N) for large datasets)
quantizer = faiss.IndexFlatL2(embedding_dim)
index_ivf = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)

print(f"Training IVF index with {nlist} clusters...")
index_ivf.train(embeddings_np)
index_ivf.add(embeddings_np)
print(f"✓ Created IndexIVFFlat with {index_ivf.ntotal} vectors")
print(f"  - Clusters vectors into {nlist} partitions")
print(f"  - Searches only nearby clusters (faster)")
print(f"  - Requires training on data")
print(f"  - Good for: 100K+ vectors")

# Set search parameters
index_ivf.nprobe = 3  # Search 3 nearest clusters
print(f"  - nprobe={index_ivf.nprobe} (higher = more accurate but slower)")

# Search
D_ivf, I_ivf = index_ivf.search(query_embedding, 3)
print("\nIVF Results:")
for i, (idx, dist) in enumerate(zip(I_ivf[0], D_ivf[0]), 1):
    print(f"  #{i} - Distance: {dist:.4f} - {tickets[idx]['ticket_id']}")

# Index Type 4: HNSW (Hierarchical Navigable Small World)
print("\n--- Index 4: IndexHNSWFlat (HNSW Graph) ---")
M = 32  # Number of connections per layer (higher = better recall but more memory)
index_hnsw = faiss.IndexHNSWFlat(embedding_dim, M)
index_hnsw.add(embeddings_np)
print(f"✓ Created IndexHNSWFlat with {index_hnsw.ntotal} vectors")
print(f"  - Builds a graph structure for fast navigation")
print(f"  - Best approximate search algorithm")
print(f"  - No training required")
print(f"  - Good for: Any size, especially 1M+ vectors")
print(f"  - Trade-off: Higher memory usage")

# Search
D_hnsw, I_hnsw = index_hnsw.search(query_embedding, 3)
print("\nHNSW Results:")
for i, (idx, dist) in enumerate(zip(I_hnsw[0], D_hnsw[0]), 1):
    print(f"  #{i} - Distance: {dist:.4f} - {tickets[idx]['ticket_id']}")

# Index Type 5: Product Quantization (Compressed)
print("\n--- Index 5: IndexIVFPQ (IVF + Product Quantization) ---")
m = 8  # Number of subquantizers (embedding_dim must be divisible by m)
nbits = 8  # Bits per subquantizer
index_ivfpq = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, m, nbits)

print(f"Training IVFPQ index...")
index_ivfpq.train(embeddings_np)
index_ivfpq.add(embeddings_np)
index_ivfpq.nprobe = 3
print(f"✓ Created IndexIVFPQ with {index_ivfpq.ntotal} vectors")
print(f"  - Compresses vectors using quantization")
print(f"  - Much lower memory usage (~{m * nbits / 32}x of original)")
print(f"  - Slightly lower accuracy")
print(f"  - Good for: Very large datasets (10M+ vectors)")

# Search
D_ivfpq, I_ivfpq = index_ivfpq.search(query_embedding, 3)
print("\nIVFPQ Results:")
for i, (idx, dist) in enumerate(zip(I_ivfpq[0], D_ivfpq[0]), 1):
    print(f"  #{i} - Distance: {dist:.4f} - {tickets[idx]['ticket_id']}")

# ============================================================================
# PART 3: Performance Benchmarking
# ============================================================================
print("\n" + "="*80)
print("PART 3: Performance Benchmarking")
print("="*80)

test_queries = [
    "database connection timeout",
    "authentication failed",
    "payment processing error",
    "mobile app crash",
    "email not delivered"
]

test_response = client.embeddings.create(input=test_queries, model=embedding_model)
test_embeddings = np.array([data.embedding for data in test_response.data], dtype='float32')
test_normalized = test_embeddings.copy()
faiss.normalize_L2(test_normalized)

indexes = [
    ("IndexFlatL2", index_flat_l2, test_embeddings),
    ("IndexFlatIP", index_flat_ip, test_normalized),
    ("IndexIVFFlat", index_ivf, test_embeddings),
    ("IndexHNSWFlat", index_hnsw, test_embeddings),
    ("IndexIVFPQ", index_ivfpq, test_embeddings),
]

print(f"\nBenchmarking {len(test_queries)} queries on each index...\n")
print(f"{'Index Type':<20} {'Avg Time (ms)':<15} {'Memory Type':<15}")
print("-" * 50)

for name, idx, test_emb in indexes:
    times = []
    for query_emb in test_emb:
        start = time.time()
        D, I = idx.search(query_emb.reshape(1, -1), 5)
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    
    if "IVF" in name:
        memory = "Medium-Low"
    elif "PQ" in name:
        memory = "Very Low"
    elif "HNSW" in name:
        memory = "High"
    else:
        memory = "Medium"
    
    print(f"{name:<20} {avg_time:<15.3f} {memory:<15}")

# ============================================================================
# PART 4: Accuracy vs Speed Trade-off
# ============================================================================
print("\n" + "="*80)
print("PART 4: Accuracy vs Speed Trade-off")
print("="*80)

# Use first test query
test_query_emb = test_embeddings[0].reshape(1, -1)

# Get ground truth from flat index
D_truth, I_truth = index_flat_l2.search(test_query_emb, 5)
truth_set = set(I_truth[0])

print(f"\nQuery: '{test_queries[0]}'")
print(f"Ground truth (exact search): {list(I_truth[0])}\n")

# Check recall for approximate indexes
approximate_indexes = [
    ("IndexIVFFlat (nprobe=1)", index_ivf, 1),
    ("IndexIVFFlat (nprobe=3)", index_ivf, 3),
    ("IndexIVFFlat (nprobe=5)", index_ivf, 5),
    ("IndexHNSWFlat", index_hnsw, None),
]

print(f"{'Index Configuration':<30} {'Recall@5':<12} {'Speed':<10}")
print("-" * 52)

for name, idx, param in approximate_indexes:
    if param is not None and hasattr(idx, 'nprobe'):
        idx.nprobe = param
    
    start = time.time()
    D, I = idx.search(test_query_emb, 5)
    elapsed = (time.time() - start) * 1000
    
    result_set = set(I[0])
    recall = len(truth_set & result_set) / len(truth_set)
    
    print(f"{name:<30} {recall:<12.1%} {elapsed:<10.3f}ms")

# ============================================================================
# PART 5: Choosing the Right Index
# ============================================================================
print("\n" + "="*80)
print("PART 5: Choosing the Right Index")
print("="*80)

print("""
Decision Guide:
--------------

Dataset Size: < 10K vectors
  → Use: IndexFlatL2 or IndexFlatIP
  → Reason: Fast enough for exact search
  → Memory: 4 * embedding_dim * N bytes

Dataset Size: 10K - 100K vectors
  → Use: IndexFlatL2/IP or IndexHNSWFlat
  → Reason: HNSW provides speed boost without much complexity
  → Memory: Moderate

Dataset Size: 100K - 1M vectors
  → Use: IndexIVFFlat or IndexHNSWFlat
  → Reason: Need approximate search, HNSW usually better
  → Memory: Medium (IVF) or High (HNSW)

Dataset Size: 1M+ vectors
  → Use: IndexHNSWFlat or IndexIVFPQ
  → Reason: HNSW for speed, IVFPQ for memory constraints
  → Memory: IVFPQ uses ~10-30x less memory

Need 100% Accuracy?
  → Use: IndexFlatL2 or IndexFlatIP only
  → Note: All approximate indexes may miss some results

Memory Constrained?
  → Use: IndexIVFPQ or IndexIVFScalarQuantizer
  → Note: Accepts some accuracy loss for huge memory savings

Real-time Search (<10ms)?
  → Use: IndexHNSWFlat or IndexIVFFlat with low nprobe
  → Note: Pre-warm the index with queries
""")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. Exact search (Flat) is best for small datasets")
print("2. HNSW is the best general-purpose approximate index")
print("3. IVF indexes are good when memory is more plentiful")
print("4. PQ compression is essential for massive datasets")
print("5. Always benchmark on your actual data and query patterns")
