"""
app.py

A Streamlit application to:
1. Search NeurIPS papers stored in ChromaDB (using embeddings).
2. Display top results with metadata and partial abstracts (with an option to download).
3. Visualize papers in t-SNE space, color-coded by category with optional category filtering,
   but with a scrollable multi-select so it doesn't flood the UI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

# Import your query function (if installed as a package, do from paper_search.scripts.query_engine import ...)
from scripts.query_engine import search_papers


TSNE_CSV_2D = "papers_with_tsne_2d.csv"  # 2D coordinates
TSNE_CSV_3D = "papers_with_tsne.csv"     # 3D coordinates

@st.cache_data
def load_tsne_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def main():
    st.set_page_config(page_title="NeurIPS Paper Explorer", layout="wide")

    # -- Inject custom CSS to limit multi-select height --
    st.markdown("""
    <style>
    /* Limit the height of the multi-select dropdown so it scrolls */
    div[data-baseweb="select"] {
        max-height: 200px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("NeurIPS Paper Explorer")
    st.write("""
    Welcome to the **NeurIPS Paper Explorer**! 
    You can **search** for relevant papers by query, or **visualize** them in 2D/3D t-SNE space. 
    """)

    st.sidebar.title("Navigation")
    page_options = ["Search Papers", "t-SNE Visualization"]
    selected_page = st.sidebar.radio("Go to", page_options)

    # ----------------- SEARCH PAGE --------------------
    if selected_page == "Search Papers":
        st.header("Search for NeurIPS Papers")
        query = st.text_input("Enter a query (e.g., 'adversarial examples in deep learning'):")
        top_k = st.slider("Number of results to return", 1, 20, 10)

        if st.button("Search"):
            if query.strip():
                with st.spinner("Searching..."):
                    results = search_papers(query, top_k=top_k)

                st.subheader(f"Top {len(results)} results for: {query}")
                if results:
                    results_data = []
                    for i, paper in enumerate(results, start=1):
                        sim_str = f"Similarity: {paper['similarity']:.4f}"
                        snippet = paper["abstract"][:300] + "..." if paper["abstract"] else "No abstract"

                        st.markdown(f"""
                        **{i}.** **[{paper['title']}]({paper['url']})**  
                        *{paper['authors']}*  
                        {sim_str}  
                        {snippet}
                        """)
                        st.write("---")

                        results_data.append({
                            "Rank": i,
                            "Title": paper["title"],
                            "Authors": paper["authors"],
                            "URL": paper["url"],
                            "Similarity": paper["similarity"],
                            "AbstractSnippet": snippet
                        })

                    # Download the results as CSV
                    df_download = pd.DataFrame(results_data)
                    csv_buffer = StringIO()
                    df_download.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download search results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name="search_results.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No results found. Try a different query.")
            else:
                st.error("Please enter a valid query.")

    # ----------------- T-SNE VISUALIZATION PAGE --------------------
    elif selected_page == "t-SNE Visualization":
        st.header("t-SNE Visualization of Papers")

        dim_option = st.radio("Choose t-SNE dimensionality:", ["2D", "3D"])
        csv_path = TSNE_CSV_3D if dim_option == "3D" else TSNE_CSV_2D

        if st.button(f"Load {dim_option} Data"):
            st.info(f"Loading t-SNE data from: {csv_path}")
            try:
                df = load_tsne_data(csv_path)
            except FileNotFoundError:
                st.error(f"Could not find file: {csv_path}")
                return

            # Make sure category is present
            if "category" not in df.columns:
                st.error("No 'category' column found in the CSV. Make sure you've assigned categories.")
                return

            # Let the user filter categories
            all_cats = sorted(df["category"].unique())

            # We'll put the multi-select in an expander to keep it tidy
            with st.expander("Filter by category (optional)", expanded=False):
                selected_cats = st.multiselect(
                    "Pick categories to display:",
                    all_cats,
                    default=all_cats
                )

            df_filtered = df[df["category"].isin(selected_cats)].copy()

            # 2D or 3D
            if dim_option == "2D":
                # Needs x, y
                fig_2d = px.scatter(
                    df_filtered,
                    x="x",
                    y="y",
                    color="category",
                    hover_data=["title", "url"],
                    title="2D t-SNE of NeurIPS Papers",
                )
                st.plotly_chart(fig_2d, use_container_width=True)

            else:
                # Needs x, y, z
                if "z" not in df_filtered.columns:
                    st.error("No 'z' column found for 3D. Check your CSV.")
                    return

                fig_3d = px.scatter_3d(
                    df_filtered,
                    x="x", y="y", z="z",
                    color="category",
                    hover_data=["title", "url"],
                    title="3D t-SNE of NeurIPS Papers"
                )
                fig_3d.update_traces(marker=dict(size=3))
                st.plotly_chart(fig_3d, use_container_width=True)

            st.success("Done! See the plot above.")
        else:
            st.write("Click the button above to load t-SNE data.")

if __name__ == "__main__":
    main()
