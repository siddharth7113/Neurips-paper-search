"""
ingest_data.py

Ingest NeurIPS papers into a local ChromaDB instance.
"""

import os
import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

def sanitize_metadata(metadata: dict) -> dict:
    """
    Safely extract metadata fields, ensuring fallback defaults.
    """
    return {
        "title": metadata.get("title", "Untitled"),
        "authors": metadata.get("authors", "Unknown") or "Unknown",
        "url": metadata.get("url", "No URL") or "No URL"
    }

def ingest_neurips_data():
    """
    Loads the NeurIPS papers JSON, embeds and ingests them into ChromaDB.
    Skips duplicates that already exist (identified by title as ID).
    """

    # 1. Path to JSON file (assuming it's in ../data/neurips_papers_last10years.json)
    json_file_path = os.path.join("data", "neurips_papers_last10years.json")
    if not os.path.isfile(json_file_path):
        raise FileNotFoundError(f"Cannot find JSON file at {json_file_path}")

    # 2. Read all the papers from JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers from JSON.")

    # 3. Initialize Chroma PersistentClient
    #    This creates/uses a directory called 'chromadb' in your project folder.
    client = PersistentClient(path="chromadb")
    collection_name = "neurips_papers"
    collection = client.get_or_create_collection(collection_name)

    # Check how many docs are already in the collection
    initial_count = collection.count()
    print(f"Collection '{collection_name}' initially has {initial_count} documents.")

    # 4. Initialize embedding model
    #    The model download may take a moment the first time you run this.
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # 5. Filter valid papers (some might lack abstract or title)
    valid_papers = [p for p in papers if p.get("abstract") and p.get("title")]
    print(f"Number of valid papers (with title & abstract): {len(valid_papers)}")

    # 6. Batch size for insertion. Adjust based on memory/performance.
    batch_size = 1000

    # 7. Process papers in batches
    for start_idx in range(0, len(valid_papers), batch_size):
        batch = valid_papers[start_idx:start_idx + batch_size]
        batch_docs = []
        batch_metas = []
        batch_embs = []
        batch_ids = []

        for paper in batch:
            paper_id = paper["title"]  # We'll use the 'title' as the unique ID

            # Check if paper already exists in Chroma by ID
            existing = collection.get(ids=[paper_id])
            if existing["ids"]:
                # If we already have this ID, skip it
                continue

            # Sanitize metadata
            metadata = sanitize_metadata({
                "title": paper["title"],
                "authors": paper.get("authors"),
                "url": paper.get("url")
            })

            # Prepare the text to embed: title + abstract
            combined_text = f"{paper['title']} {paper['abstract']}"
            embedding = model.encode(combined_text)

            # Add them to batch arrays
            batch_docs.append(paper["abstract"])   # document text
            batch_metas.append(metadata)           # cleaned metadata
            batch_embs.append(embedding)           # embedding vector
            batch_ids.append(paper_id)             # unique ID

        # If we have new papers in this batch, add them to Chroma
        if batch_docs:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embs,
                ids=batch_ids
            )
            print(f"Ingested {len(batch_docs)} papers in this batch (from index {start_idx}).")
        else:
            print(f"No new papers to ingest in batch starting at index {start_idx}.")

    # 8. Final count
    final_count = collection.count()
    print(f"Done. Collection '{collection_name}' now has {final_count} documents total.")
    new_count = final_count - initial_count
    print(f"Ingested a total of {new_count} new documents.")

if __name__ == "__main__":
    ingest_neurips_data()
