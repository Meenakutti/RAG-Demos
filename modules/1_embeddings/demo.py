# -*- coding: utf-8 -*-
"""
Hour 1: Embeddings & Similarity Search Demo
============================================

This demo teaches:
1. How to generate embeddings from text
2. Computing similarity scores
3. Finding most similar documents
4. Visualizing embeddings in 2D space

LEARNING RESOURCES:
- OpenAI Embeddings Guide: https://platform.openai.com/docs/guides/embeddings
- Understanding Vector Embeddings: https://www.pinecone.io/learn/vector-embeddings/
- Cosine Similarity Explained: https://en.wikipedia.org/wiki/Cosine_similarity
- Semantic Search Intro: https://www.sbert.net/examples/applications/semantic-search/README.html
"""

import json
import numpy as np  # For numerical operations on embedding vectors
import os
from openai import OpenAI  # OpenAI API client for generating embeddings
from sklearn.metrics.pairwise import cosine_similarity  # Measure similarity between vectors
import matplotlib.pyplot as plt  # For visualizing embeddings
from sklearn.decomposition import PCA  # Dimensionality reduction for visualization
from dotenv import load_dotenv  # Load environment variables from .env file

# Load environment variables (API keys, model names, etc.)
# Best practice: Never hardcode API keys in your code!
load_dotenv()

# Initialize OpenAI client
# Reference: https://platform.openai.com/docs/api-reference/embeddings
print("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Choose embedding model
# text-embedding-3-small: 1536 dimensions, fast and cost-effective
# text-embedding-3-large: 3072 dimensions, higher quality but more expensive
# Reference: https://platform.openai.com/docs/guides/embeddings/embedding-models
embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
embedding_dim = 1536  # Number of dimensions in the embedding vector
print(f"Using OpenAI model: {embedding_model}")
print(f"Embedding dimension: {embedding_dim}")

# Load synthetic tickets
print("\nLoading support tickets...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"Loaded {len(tickets)} support tickets")

# Display sample ticket
print("\n" + "="*80)
print("SAMPLE TICKET:")
print("="*80)
sample = tickets[0]
print(f"ID: {sample['ticket_id']}")
print(f"Title: {sample['title']}")
print(f"Description: {sample['description'][:200]}...")
print(f"Category: {sample['category']}")
print(f"Priority: {sample['priority']}")

# ============================================================================
# PART 1: Generate Embeddings
# ============================================================================
print("\n" + "="*80)
print("PART 1: Generating Embeddings")
print("="*80)

# Combine title and description for richer context
# TIP: More context generally leads to better embeddings
# Include all relevant information that helps distinguish this document from others
ticket_texts = [
    f"{ticket['title']}. {ticket['description']}" 
    for ticket in tickets
]

# Generate embeddings using OpenAI's API
# What are embeddings? High-dimensional vectors that capture semantic meaning
# Similar meanings → similar vectors (measured by distance/angle between vectors)
# Reference: https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
print("\nGenerating embeddings for all tickets...")
response = client.embeddings.create(input=ticket_texts, model=embedding_model)

# Convert API response to NumPy array for easier mathematical operations
embeddings = np.array([data.embedding for data in response.data])
print(f"✓ Generated embeddings with shape: {embeddings.shape}")
print(f"  ({len(tickets)} tickets × {embedding_dim} dimensions)")

# Show what an embedding looks like
# Each number represents the "strength" along a semantic dimension
# You can't interpret individual values, but the overall pattern captures meaning
print(f"\nFirst 10 values of embedding for ticket 1:")
print(embeddings[0][:10])
print("  (These numbers encode the semantic meaning of the text)")

# ============================================================================
# PART 2: Compute Similarity Scores
# ============================================================================
print("\n" + "="*80)
print("PART 2: Computing Similarity Scores")
print("="*80)

# Create a search query
query = "Users can't authenticate after changing password"
print(f"\nSearch Query: '{query}'")

# Generate embedding for the query using the SAME model as documents
# IMPORTANT: Always use the same embedding model for queries and documents!
# Different models produce incompatible vector spaces
query_response = client.embeddings.create(input=[query], model=embedding_model)
query_embedding = np.array([query_response.data[0].embedding])
print(f"Query embedding shape: {query_embedding.shape}")

# Compute cosine similarity between query and all tickets
# Cosine similarity measures the angle between vectors (range: -1 to 1)
# 1 = identical direction (very similar)
# 0 = perpendicular (unrelated)
# -1 = opposite direction (contradictory)
# Reference: https://en.wikipedia.org/wiki/Cosine_similarity
# Why cosine? It's invariant to vector magnitude, only cares about direction
similarities = cosine_similarity(query_embedding, embeddings)[0]
print(f"\nComputed similarity scores for {len(similarities)} tickets")
print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")

# ============================================================================
# PART 3: Retrieve Most Similar Tickets
# ============================================================================
print("\n" + "="*80)
print("PART 3: Finding Most Similar Tickets")
print("="*80)

# Get top-5 most similar tickets
# This is the core of semantic search: rank by similarity score
top_k = 5

# np.argsort returns indices that would sort the array
# [::-1] reverses to get descending order (highest similarity first)
# [:top_k] takes only the top K results
top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"\nTop {top_k} most similar tickets to query: '{query}'")
print("-" * 80)

for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]
    
    print(f"\n#{rank} - Similarity: {score:.4f}")
    print(f"Ticket ID: {ticket['ticket_id']}")
    print(f"Title: {ticket['title']}")
    print(f"Category: {ticket['category']} | Priority: {ticket['priority']}")
    print(f"Description: {ticket['description'][:150]}...")

# ============================================================================
# PART 4: Visualize Embeddings in Semantic Space
# ============================================================================
print("\n" + "="*80)
print("PART 4: Visualizing Embeddings in Semantic Space")
print("="*80)

print(f"\nTo visualize {embedding_dim}-dimensional embeddings, we need to project them into 2D...")
print("Think of it like taking a photo of a 3D object - we lose some detail but can see relationships.")

# Use PCA (Principal Component Analysis) to reduce dimensions
# PCA finds the 2 directions that capture the most variance in the data
# Reference: https://scikit-learn.org/stable/modules/decomposition.html#pca
# Don't worry about the math - just know it preserves relative distances as much as possible
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)  # Transform all ticket embeddings
query_2d = pca.transform(query_embedding)  # Transform query using same projection

print("✓ Embeddings projected to 2D for visualization")
print("\nWhat you'll see:")
print("  • Each dot = one support ticket")
print("  • Closer dots = more semantically similar tickets")
print("  • Red star = your search query")
print("  • Red circles = top-5 matches")

# Create the plot
# Matplotlib reference: https://matplotlib.org/stable/tutorials/index.html
plt.figure(figsize=(12, 8))

# Plot all tickets by category
categories = list(set(ticket['category'] for ticket in tickets))
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
category_to_color = dict(zip(categories, colors))

for i, ticket in enumerate(tickets):
    color = category_to_color[ticket['category']]
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
               c=[color], label=ticket['category'], s=100, alpha=0.6)

# Highlight top-5 matches with red circles
for idx in top_indices:
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
               s=300, facecolors='none', edgecolors='red', linewidths=2)

# Plot query as red star
plt.scatter(query_2d[0, 0], query_2d[0, 1], 
           c='red', marker='*', s=500, label='Query', edgecolors='black', linewidths=2)

# Remove duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best')

plt.title('Embeddings in Semantic Space: Similar Meanings Cluster Together', 
         fontsize=14, fontweight='bold')
plt.xlabel('Semantic Dimension 1')
plt.ylabel('Semantic Dimension 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('embeddings_visualization.png', dpi=150)
print("\n✓ Visualization saved as 'embeddings_visualization.png'")
plt.show()

# ============================================================================
# PART 5: Experiment with Different Queries
# ============================================================================
print("\n" + "="*80)
print("PART 5: Try Different Queries")
print("="*80)

test_queries = [
    "Database is timing out",
    "Payment not working for foreign customers",
    "App crashes on iPhone",
    "Emails are not being sent"
]

print("\nTesting semantic search with different queries:")
for test_query in test_queries:
    query_resp = client.embeddings.create(input=[test_query], model=embedding_model)
    query_emb = np.array([query_resp.data[0].embedding])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argmax(sims)
    
    print(f"\nQuery: '{test_query}'")
    print(f"  → Best match: {tickets[top_idx]['title']}")
    print(f"  → Similarity: {sims[top_idx]:.4f}")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. Embeddings convert text into numerical vectors")
print("2. Similar meanings → similar vectors (measured by cosine similarity)")
print("3. Semantic search finds meaning, not just keywords")
print("4. Embeddings can be visualized to understand relationships")
print("\nNext: Hour 2 - Chunking & Vector Stores")
