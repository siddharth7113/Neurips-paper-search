"""
tsne_visualization.py

1. Fetches embeddings + metadata from ChromaDB.
2. Reads assigned categories from 'paper_categories.csv'.
3. Runs t-SNE to reduce embeddings to 3D (or 2D if you like).
4. Creates a Plotly 3D scatter plot, color-coding by assigned category.
5. Saves the final DataFrame + an HTML plot.
"""

import os
import pandas as pd
from chromadb import PersistentClient
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio

def run_tsne_visualization(
    collection_name="neurips_papers",
    categories_csv="paper_categories.csv",
    output_csv="papers_with_tsne.csv",
    output_html="tsne_3d_plot.html",
    n_components=3,
    random_state=42,
    perplexity=30.0
):
    # ------------------------------
    # 1. Connect to ChromaDB & fetch embeddings
    # ------------------------------
    client = PersistentClient(path="chromadb")
    collection = client.get_or_create_collection(collection_name)
    print(f"Fetching all embeddings & metadata from collection '{collection_name}'...")

    all_data = collection.get(include=["embeddings", "metadatas"])
    embeddings = all_data["embeddings"]     # List of embedding vectors
    metadatas = all_data["metadatas"]       # List of metadata dicts

    num_papers = len(embeddings)
    print(f"Fetched {num_papers} papers from ChromaDB.")

    # ------------------------------
    # 2. Load assigned categories from CSV
    # ------------------------------
    if not os.path.isfile(categories_csv):
        raise FileNotFoundError(f"Cannot find categories CSV at {categories_csv}")

    print(f"Loading categories from: {categories_csv}")
    cat_df = pd.read_csv(categories_csv)  # columns: [title, assigned_category, url] (or similar)

    cat_df.rename(columns={"assigned_category": "category"}, inplace=True)

    # ------------------------------
    # 3. Create a DataFrame merging metadata + category
    # ------------------------------
    titles = [meta.get("title", "Untitled") for meta in metadatas]
    urls = [meta.get("url", "No URL") for meta in metadatas]

    papers_df = pd.DataFrame({
        "title": titles,
        "url": urls
    })

    # Merge
    papers_df = pd.merge(papers_df, cat_df[["title", "category"]], on="title", how="left")

    # ------------------------------
    # 4. Run t-SNE
    # ------------------------------
    print(f"Running t-SNE (n_components={n_components}, perplexity={perplexity}) on {num_papers} embeddings...")
    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    tsne_results = tsne.fit_transform(embeddings)

    if n_components == 3:
        papers_df["x"] = tsne_results[:, 0]
        papers_df["y"] = tsne_results[:, 1]
        papers_df["z"] = tsne_results[:, 2]
    elif n_components == 2:
        papers_df["x"] = tsne_results[:, 0]
        papers_df["y"] = tsne_results[:, 1]
    else:
        raise ValueError(f"Unsupported number of dimensions: {n_components}")

    print("t-SNE complete!")

    # ------------------------------
    # 5. Save the final DataFrame with t-SNE coordinates
    # ------------------------------
    print(f"Saving t-SNE results to CSV: {output_csv}")
    papers_df.to_csv(output_csv, index=False)

    # ------------------------------
    # 6. Optionally create a 3D scatter plot
    # ------------------------------
    if n_components == 3:
        pio.renderers.default = "browser"

        print("Creating 3D scatter plot...")
        fig = px.scatter_3d(
            papers_df,
            x="x",
            y="y",
            z="z",
            color="category",
            hover_data={"title": True, "url": False},
            custom_data=["title", "url"],
            title="3D t-SNE of NeurIPS Papers by Category",
            labels={"category": "Category"},
        )

        fig.update_traces(
            hovertemplate="<b>Title:</b> %{customdata[0]}<br>"
                          "<b>URL:</b> <a href='%{customdata[1]}' target='_blank'>%{customdata[1]}</a><br>"
        )
        fig.update_traces(marker=dict(size=5, opacity=0.8))

        print(f"Saving 3D Plotly figure to: {output_html}")
        fig.write_html(output_html, include_plotlyjs="cdn")

        # Optionally display locally:
        # fig.show()
    else:
        print("n_components != 3, skipping 3D plot creation.")

    print("Done! Check your CSV and HTML for results.")

def main():
    run_tsne_visualization()

if __name__ == "__main__":
    main()
