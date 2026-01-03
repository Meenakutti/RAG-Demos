# Indexing Strategies Exercises

## Exercise 1: Index Performance Comparison (Easy)

**Task**: Measure build time and query latency for different index types.

**Steps**:
```python
import time

def benchmark_index(index_builder, name, embeddings, test_queries):
    # Build index
    start = time.time()
    index = index_builder(embeddings)
    build_time = time.time() - start
    
    # Query latency
    query_times = []
    for query_emb in test_queries:
        start = time.time()
        D, I = index.search(query_emb.reshape(1, -1), 5)
        query_times.append((time.time() - start) * 1000)
    
    avg_query_time = sum(query_times) / len(query_times)
    
    print(f"\n{name}:")
    print(f"  Build time: {build_time:.3f}s")
    print(f"  Avg query time: {avg_query_time:.2f}ms")

# Test different indexes
def build_flat_l2(embeddings):
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def build_hnsw(embeddings):
    index = faiss.IndexHNSWFlat(embedding_dim, 32)
    index.add(embeddings)
    return index

# Run benchmarks
benchmark_index(build_flat_l2, "IndexFlatL2", embeddings, test_queries)
benchmark_index(build_hnsw, "IndexHNSWFlat", embeddings, test_queries)
```

**Questions**:
- Which index builds faster?
- Which index searches faster?
- What's the trade-off between build and search time?

---

## Exercise 2: IVF Parameter Tuning (Medium)

**Task**: Experiment with IVF parameters to understand the accuracy/speed trade-off.

**Parameters to test**:
```python
# Number of clusters
nlist_values = [5, 10, 20, 50]

# Number of clusters to search
nprobe_values = [1, 2, 5, 10]

for nlist in nlist_values:
    quantizer = faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
    index.train(embeddings)
    index.add(embeddings)
    
    for nprobe in nprobe_values:
        index.nprobe = nprobe
        # Measure accuracy and speed
        # ...
```

**Metrics to measure**:
1. Recall@5 (compared to exact search)
2. Query latency
3. Build time

**Analysis Questions**:
- How does `nlist` affect build time?
- How does `nprobe` affect recall?
- What's the optimal `nlist/nprobe` combination?
- Rule of thumb: nlist ≈ sqrt(N), nprobe ≈ sqrt(nlist)

---

## Exercise 3: Memory Usage Analysis (Medium)

**Task**: Calculate and compare memory usage for different index types.

**Approach**:
```python
import sys

def estimate_index_memory(index, embeddings):
    """Estimate memory usage of a FAISS index"""
    # For IndexFlat: dimension * ntotal * 4 bytes (float32)
    # For HNSW: ~(dimension * ntotal * 4) * 1.5 to 2.0
    # For IVFPQ: (m * nbits * ntotal) / 8
    
    if isinstance(index, faiss.IndexFlat):
        return (embeddings.shape[1] * index.ntotal * 4) / (1024**2)  # MB
    # Add other index types...

# Calculate for each index type
indexes = {
    'FlatL2': index_flat_l2,
    'HNSW': index_hnsw,
    'IVFPQ': index_ivfpq,
}

for name, idx in indexes.items():
    memory_mb = estimate_index_memory(idx, embeddings)
    print(f"{name}: {memory_mb:.2f} MB")
```

**Challenge**: For a dataset with 1M vectors (384-dim):
- Calculate memory for IndexFlatL2
- Calculate memory for IndexIVFPQ (m=8, nbits=8)
- How much memory do you save with compression?

---

## Exercise 4: L2 vs Inner Product (Cosine) Similarity (Medium)

**Task**: Compare L2 distance and cosine similarity for semantic search.

**Implementation**:
```python
import faiss

# L2 Distance (Euclidean)
index_l2 = faiss.IndexFlatL2(embedding_dim)
index_l2.add(embeddings)

# Inner Product (Cosine similarity with normalized vectors)
embeddings_norm = embeddings.copy()
faiss.normalize_L2(embeddings_norm)
index_ip = faiss.IndexFlatIP(embedding_dim)
index_ip.add(embeddings_norm)

# Search with both
query = "password reset issue"
query_emb = model.encode([query]).astype('float32')
query_norm = query_emb.copy()
faiss.normalize_L2(query_norm)

D_l2, I_l2 = index_l2.search(query_emb, 5)
D_ip, I_ip = index_ip.search(query_norm, 5)

print("L2 Results:", I_l2[0])
print("IP Results:", I_ip[0])
```

**Questions**:
- Do the results differ significantly?
- Which is better for semantic similarity?
- Why do we normalize for inner product?
- Hint: cos(θ) = (A·B) / (||A|| ||B||)

---

## Exercise 5: Building a Production-Ready Index (Hard)

**Task**: Implement a class that automatically selects the best index type based on dataset size.

**Requirements**:
```python
class AdaptiveVectorIndex:
    def __init__(self, embedding_dim, metric='l2'):
        """
        Auto-select index type based on dataset size
        
        Args:
            embedding_dim: Dimension of embeddings
            metric: 'l2' or 'cosine'
        """
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.index = None
        self.index_type = None
    
    def build(self, embeddings):
        """Build optimal index based on data size"""
        n_vectors = len(embeddings)
        
        if n_vectors < 10000:
            # Use exact search
            self.index_type = "Flat"
            self.index = self._build_flat(embeddings)
        elif n_vectors < 100000:
            # Use HNSW
            self.index_type = "HNSW"
            self.index = self._build_hnsw(embeddings)
        else:
            # Use IVF
            self.index_type = "IVF"
            self.index = self._build_ivf(embeddings)
        
        print(f"Built {self.index_type} index for {n_vectors} vectors")
    
    def _build_flat(self, embeddings):
        # TODO: Implement
        pass
    
    def _build_hnsw(self, embeddings):
        # TODO: Implement with optimal M parameter
        pass
    
    def _build_ivf(self, embeddings):
        # TODO: Implement with optimal nlist
        pass
    
    def search(self, query, k=5):
        """Search the index"""
        # TODO: Implement
        pass
    
    def save(self, path):
        """Save index to disk"""
        faiss.write_index(self.index, path)
    
    def load(self, path):
        """Load index from disk"""
        self.index = faiss.read_index(path)
```

**Test it**:
```python
# Small dataset
adaptive_idx = AdaptiveVectorIndex(384)
adaptive_idx.build(embeddings[:100])  # Should use Flat

# Medium dataset
adaptive_idx.build(embeddings[:50000])  # Should use HNSW

# Large dataset
adaptive_idx.build(embeddings[:500000])  # Should use IVF
```

---

## Exercise 6: Index Persistence and Loading (Easy)

**Task**: Practice saving and loading FAISS indexes.

**Steps**:
```python
import faiss

# Build an index
index = faiss.IndexHNSWFlat(embedding_dim, 32)
index.add(embeddings)

# Save to disk
faiss.write_index(index, "./my_index.faiss")
print("Index saved!")

# Load from disk
loaded_index = faiss.read_index("./my_index.faiss")
print(f"Loaded index with {loaded_index.ntotal} vectors")

# Verify it works
D, I = loaded_index.search(query_embedding, 3)
print("Search results:", I[0])
```

**Why this matters**: 
- Building indexes is expensive
- Production systems should load pre-built indexes
- Much faster startup time

---

## Exercise 7: Hybrid Index Strategy (Advanced)

**Task**: Combine exact and approximate search for best of both worlds.

**Concept**: Use approximate search to quickly find candidates, then re-rank with exact search.

**Implementation**:
```python
def hybrid_search(query_emb, index_approx, index_exact, k=5, candidate_multiplier=3):
    """
    Two-stage search: fast approximate + exact reranking
    
    Args:
        query_emb: Query embedding
        index_approx: Fast approximate index (HNSW/IVF)
        index_exact: Exact index (Flat)
        k: Final number of results
        candidate_multiplier: How many candidates to retrieve
    
    Returns:
        Top-k results with exact scores
    """
    # Stage 1: Get candidates from approximate index
    k_candidates = k * candidate_multiplier
    D_approx, I_approx = index_approx.search(query_emb, k_candidates)
    
    # Stage 2: Re-rank candidates with exact search
    # Extract candidate vectors from exact index
    # Compute exact distances
    # Return top-k
    
    # TODO: Implement reranking
    pass

# Test it
index_approx = faiss.IndexHNSWFlat(embedding_dim, 32)
index_approx.add(embeddings)

index_exact = faiss.IndexFlatL2(embedding_dim)
index_exact.add(embeddings)

results = hybrid_search(query_embedding, index_approx, index_exact, k=5)
```

**Benefits**:
- Faster than pure exact search
- More accurate than pure approximate search
- Good balance for production systems

---

## Bonus: GPU Acceleration (Advanced)

**Task**: Use GPU for faster indexing and search (if available).

```python
import faiss

# Check if GPU is available
if faiss.get_num_gpus() > 0:
    print(f"Found {faiss.get_num_gpus()} GPU(s)")
    
    # Create GPU resources
    res = faiss.StandardGpuResources()
    
    # CPU index
    cpu_index = faiss.IndexFlatL2(embedding_dim)
    
    # Move to GPU
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(embeddings)
    
    # Search on GPU (much faster!)
    D, I = gpu_index.search(query_embedding, 5)
else:
    print("No GPU available")
```

**Performance gain**: 10-100x faster for large datasets!

---

## Solutions

Solutions provided after the workshop. Try to implement these yourself first!

---

## Next Steps

Ready for **RAG Pipeline**? We'll combine embeddings, chunking, and indexing into a complete question-answering system!

---

**Need help?** Ask the instructor or check the demo code for reference implementations.
