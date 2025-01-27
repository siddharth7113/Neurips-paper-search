from setuptools import setup, find_packages

setup(
    name="paper_search",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "chromadb>=0.3.21",
        "sentence-transformers>=2.2.2",
        "scikit-learn>=1.2.2",
        "plotly>=5.13.1",
        "streamlit>=1.20.0",
        "pandas>=1.5.2",
    ],
    python_requires=">=3.12",
)
