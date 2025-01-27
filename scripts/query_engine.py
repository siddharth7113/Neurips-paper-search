#!/usr/bin/env python3

"""
query_engine.py (Refined)

Provides functions to search for top K most similar papers from the ChromaDB
'neurips_papers' collection given a natural-language query.

Key Features:
- Caches the SentenceTransformer model to avoid reloading it on every search
  (useful if you're calling `search_papers()` multiple times in an app).
- Handles ambiguous distance measures (cosine vs. L2) by auto-adjusting:
  1) If distance <= 1, we assume 'cosine distance' => similarity = 1 - distance
  2) If distance > 1, we assume 'L2 distance' => similarity = 1 / (1 + distance)
- Prints refined console output with properly formatted similarity scores.
"""

import sys
from typing import List, Dict, Any

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

_MODEL_CACHE = None  # Global cache for the embedding model

def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Returns a cached SentenceTransformer model to avoid re-initialization overhead.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        print(f"[query_engine] Loading SentenceTransformer model: {model_name}")
        _MODEL_CACHE = SentenceTransformer(model_name)
    return _MODEL_CACHE

def _compute_similarity_from_distance(distance: float) -> float:
    """
    Attempt to interpret the distance as either:
    - Cosine distance in [0, 2] => similarity = 1 - distance (only if distance <= 1.0)
    - Euclidean or other distance => similarity = 1 / (1 + distance)

    This avoids negative similarities if distance > 1 under a supposed "cosine" interpretation.
    Feel free to refine this logic based on your actual metric.
    """
    if distance <= 1.0:
        # Probably a true "cosine distance" in [0,1]
        similarity = 1.0 - distance
    else:
        # Possibly an L2 distance or bigger than 1 => do a generic inverse transformation
        similarity = 1.0 / (1.0 + distance)

    return similarity


def search_papers(
    query: str,
    top_k: int = 10,
    collection_name: str = "neurips_papers",
    model_name: str = "all-MiniLM-L6-v2"
) -> List[Dict[str, Any]]:
    """
    Embeds the user query using a SentenceTransformer model,
    and performs a similarity query on the 'neurips_papers' collection
    in ChromaDB. Returns a list of result dicts:

    [
      {
        'title': str,
        'authors': str,
        'url': str,
        'abstract': str,
        'similarity': float
      },
      ...
    ]

    Args:
        query (str): The user's search query (natural language).
        top_k (int): Number of results to return.
        collection_name (str): Name of the ChromaDB collection to query.
        model_name (str): The SentenceTransformer model to use.

    Returns:
        A list of dictionaries with paper info, sorted by similarity descending.
    """
    if not query.strip():
        # If the query is empty or just whitespace, return an empty list.
        return []

    # 1. Connect to Chroma
    client = PersistentClient(path="chromadb")
    collection = client.get_or_create_collection(collection_name)

    # 2. Load (or retrieve) the cached embedding model
    model = _get_embedding_model(model_name)

    # 3. Embed the query
    query_embedding = model.encode(query)

    # 4. Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 5. Format the result list
    papers = []
    if results and results["metadatas"]:
        result_metadatas = results["metadatas"][0]
        result_docs = results["documents"][0]
        result_dists = results["distances"][0]

        for meta, doc, dist in zip(result_metadatas, result_docs, result_dists):
            title = meta.get("title", "Untitled")
            authors = meta.get("authors", "Unknown")
            url = meta.get("url", "No URL")

            # Convert the distance to a "similarity" measure
            similarity = _compute_similarity_from_distance(dist)

            paper_info = {
                "title": title,
                "authors": authors,
                "url": url,
                "abstract": doc,
                "similarity": similarity
            }
            papers.append(paper_info)

        # Sort by similarity descending
        papers.sort(key=lambda x: x["similarity"], reverse=True)

    return papers

def main():
    """
    Simple command-line interface for testing.
    Usage:
        python scripts/query_engine.py
    """
    # If the user typed a query as a CLI argument, e.g.:
    #    python scripts/query_engine.py "adversarial examples"
    # we can parse sys.argv. Otherwise we prompt.
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your query: ").strip()

    top_k = 5
    results = search_papers(user_query, top_k=top_k)

    print(f"\nTop {len(results)} results for query: '{user_query}'\n")
    for i, p in enumerate(results, start=1):
        print(f"{i}. Title: {p['title']}")
        print(f"   Authors: {p['authors']}")
        print(f"   URL: {p['url']}")
        print(f"   Similarity: {p['similarity']:.4f}")  # 4 decimal places
        # Truncate abstract for readability
        snippet = (p['abstract'][:200] + "...") if p['abstract'] else "No abstract"
        print(f"   Abstract: {snippet}\n")

if __name__ == "__main__":
    main()
