#!/usr/bin/env python3
"""
categorize_papers.py

Assigns each paper a category by comparing its embedding to known category-description embeddings.
Stores results in a CSV or updates the DB metadata.
"""

import os
import csv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def categorize_papers():
    # 1. Define category descriptions
    category_descriptions = {
        "Deep Learning->Everything Else": "Deep learning models, neural networks, and general model training techniques.",
        "Optimization->Convex": "Mathematical optimization focusing on convex problems and linear programming.",
        "Privacy": "Privacy, secure data handling, and confidentiality techniques in machine learning.",
        "Theory->Active Learning and Interactive Learning": "Active learning strategies and interactive methods to enhance model training.",
        "Multimodal Models": "Models that integrate and process multiple types of data or sensors, enabling cross-modal understanding.",
        "Physical Models->Physics": "Modeling physical systems, simulations, and applications of physics in machine learning.",
        "Language->Generation": "Techniques and models for generating natural language, including text and language generation.",
        "Physical Models": "General physical models, simulations, and mechanics applied within machine learning contexts.",
        "Generative Models->In Context Learning": "In-context learning approaches and prompt-based learning within generative models.",
        "Fairness": "Ensuring fairness, mitigating bias, and promoting ethical AI practices in machine learning.",
        "Reinforcement Learning->Everything Else": "General reinforcement learning topics and methodologies not covered in specific subcategories.",
        "Generative Models": "Models that generate data, including GANs, autoencoders, and other generative architectures.",
        "Language": "Natural language processing, linguistics, and general language-related machine learning tasks.",
        "Neuroscience, Cognitive Science": "Integrating neuroscience and cognitive science principles with machine learning for understanding brain and behavior.",
        "Computer Vision": "Image processing, object detection, and general computer vision applications in machine learning.",
        "Deep Learning->Theory": "Theoretical foundations and analyses of deep learning algorithms and architectures.",
        "Generative Models->Diffusion models": "Diffusion-based generative models and denoising techniques in data generation.",
        "Datasets and Benchmarks": "Collection, curation, and evaluation of datasets and benchmarks for machine learning research.",
        "Trustworthy Machine Learning": "Developing robust, reliable, and secure AI systems that can be trusted in various applications.",
        "Computer Vision->Image Generation": "Generating images and vision synthesis using advanced computer vision techniques.",
        "Learning Theory": "Theoretical aspects of machine learning, including convergence, generalization, and learning guarantees.",
        "Optimization->Discrete and Combinatorial Optimization": "Solving discrete and combinatorial problems using optimization techniques.",
        "Theory->Statistics": "Statistical methods and probabilistic models underpinning machine learning algorithms.",
        "Robotics": "Integration of machine learning in robotics, including control, autonomy, and robot motion planning.",
        "Data-centric AI": "Approaches focusing on data quality, data efficiency, and data-centric methodologies in AI.",
        "Causality": "Causal inference, cause-effect relationships, and causality-based machine learning models.",
        "Interpretability": "Techniques for making machine learning models interpretable and explainable to humans.",
        "Probabilistic Methods->Bayesian Models and Methods": "Bayesian approaches and probabilistic inference methods in machine learning.",
        "Optimization->Learning for Optimization": "Using machine learning to improve optimization processes, including meta-optimization techniques.",
        "Reinforcement Learning->Planning": "Planning algorithms and decision-making strategies within reinforcement learning frameworks.",
        "Applications": "Practical applications and use cases of machine learning across various industries and domains.",
        "Theory->Game Theory": "Application of game theory and strategic interactions in machine learning contexts.",
        "Reinforcement Learning->Multi-agent": "Multi-agent reinforcement learning, including cooperative and competitive scenarios.",
        "Graph Neural Networks": "Neural network architectures designed to work with graph-structured data and representations.",
        "Deep Learning->Algorithms": "Algorithms specific to deep learning, including model updates and training procedures.",
        "Computational Biology": "Application of machine learning in genomics, proteomics, bioinformatics, and biomedical sciences.",
        "Miscellaneous Aspects of Machine Learning->Transfer, Multitask and Meta-learning": "Learning techniques that transfer knowledge across tasks, multitask learning, and meta-learning strategies.",
        "Time Series": "Analysis and forecasting of sequential and time-dependent data using machine learning models.",
        "Miscellaneous Aspects of Machine Learning->Supervised Learning": "Supervised learning techniques, including classification and regression methods.",
        "Statistical Methods": "Statistical approaches and probabilistic methods used in developing machine learning models.",
        "Theory->Reinforcement Learning and Planning": "Theoretical foundations of reinforcement learning and planning algorithms.",
        "Optimization": "General optimization techniques, solvers, and approaches to tackle optimization problems in machine learning.",
        "Theory->Everything Else": "Foundational theories, mathematical principles, and miscellaneous theoretical aspects of machine learning.",
        "Computer Vision->Video Understanding": "Understanding and analyzing video data, including action recognition and temporal dynamics.",
        "Optimization->Optimization and Learning under Uncertainty": "Optimization techniques that account for uncertainty and stochastic elements in learning.",
        "Deep Learning->Representation Learning": "Learning meaningful representations and features from data using deep learning models.",
        "Generative Models->Misc": "Miscellaneous topics related to generative models that do not fit into other subcategories.",
        "Probabilistic Methods": "Probabilistic approaches and random processes in machine learning methodologies.",
        "Computer Vision->Classification": "Image classification tasks and techniques within computer vision.",
        "Online Learning": "Learning algorithms that update models incrementally as new data arrives.",
        "Generative Models->Reasoning": "Incorporating reasoning and logical structures within generative models.",
        "Optimization->Non-Convex": "Handling non-convex optimization problems and complex optimization landscapes in machine learning.",
        "Reinforcement Learning->Deep RL": "Deep reinforcement learning approaches that combine deep learning with RL methodologies.",
        "Computer Vision->Video Generation": "Generating video content and video synthesis using advanced computer vision techniques.",
        "Probabilistic Methods->Tractable Models": "Developing and utilizing probabilistic models that are computationally tractable.",
        "Miscellaneous Aspects of Machine Learning->General Machine Learning Techniques": "Fundamental machine learning techniques and basic concepts applicable across various domains.",
        "Deep Learning->Attention Mechanisms": "Attention-based mechanisms, including transformers and self-attention in deep learning models.",
        "Deep Learning->Robustness": "Enhancing the robustness of deep learning models against adversarial attacks and perturbations.",
        "Computer Vision->Segmentation": "Image segmentation techniques and applications within computer vision.",
        "Miscellaneous Aspects of Machine Learning->Unsupervised and Semi-supervised Learning": "Learning methods that do not rely solely on labeled data, including unsupervised and semi-supervised approaches.",
        "Probabilistic Methods->Variational Inference": "Variational inference techniques and methods for approximate probabilistic modeling.",
        "Computer Vision->Stereo": "Stereo vision and 3D vision techniques in computer vision applications.",
        "Reinforcement Learning->Batch Offline": "Batch reinforcement learning and offline RL methodologies.",
        "Computer Vision->Text Understanding": "Understanding and processing text within images, including OCR and related tasks.",
        "Probabilistic Methods->Gaussian Processes": "Gaussian processes and their applications in probabilistic modeling and machine learning.",
        "Human-computer interaction": "Interfacing machine learning models with users, focusing on user interaction and interface design.",
        "Language->Factuality": "Ensuring factual accuracy and truthfulness in generated text and language models.",
        "Probabilistic Methods->Monte Carlo and Sampling Methods": "Monte Carlo simulations and sampling techniques in probabilistic machine learning.",
        "Reinforcement Learning->Function Approximation": "Approaches for approximating functions within reinforcement learning frameworks.",
        "Language->Speech": "Speech recognition (ASR), text-to-speech (TTS), and other speech-related machine learning applications.",
        "Deep Learning->Self-Supervised Learning": "Self-supervised learning techniques and unsupervised pretraining methods in deep learning.",
        "Physical Models->Climate": "Climate modeling, weather prediction, and applications of machine learning to climate change studies.",
        "Generative Models->New Approaches": "Novel and innovative methods in the development of generative models.",
        "Data-centric AI->Data-centric AI methods and tools": "Tools and methodologies focused on enhancing data-centric AI approaches.",
        "Optimization->Bilevel optimization": "Bilevel optimization techniques and nested optimization problems in machine learning.",
        "Theory->Probabilistic Methods": "Theoretical aspects of probabilistic methods and their foundations in machine learning.",
        "Optimization->Large Scale, Parallel and Distributed": "Optimization techniques designed for large-scale, parallel, and distributed computing environments.",
        "Optimization->Zero-order and Black-box Optimization": "Optimization methods that do not require gradient information, including zero-order and black-box approaches.",
        "Language->Knowledge": "Knowledge representation, knowledge graphs, and integrating knowledge into language models.",
        "Theory->Domain Adaptation and Transfer Learning": "Theoretical foundations of domain adaptation and transfer learning in machine learning.",
        "Data-centric AI->Data augmentation": "Techniques for augmenting and enhancing data to improve machine learning model performance.",
        "Miscellaneous Aspects of Machine Learning->Representation Learning": "Learning representations that capture the underlying structure of data in various machine learning contexts.",
        "Language->Dialogue": "Conversational AI, dialogue systems, and models designed for interactive language understanding.",
        "Social Aspects": "Social impact, ethical considerations, and the broader societal implications of machine learning technologies.",
        "Miscellaneous Aspects of Machine Learning->Kernel methods": "Kernel-based methods, including Support Vector Machines (SVM) and related algorithms.",
        "Deep Learning->Autoencoders": "Autoencoder architectures and their applications in dimensionality reduction and feature learning.",
        "Miscellaneous Aspects of Machine Learning": "General machine learning topics that encompass a wide range of subjects not covered by specific categories.",
        "Physical Models->Geoscience": "Applications of machine learning in geoscience, earth science, and related environmental studies."
    }

    # 2. Initialize Chroma and fetch paper embeddings
    client = PersistentClient(path="chromadb")
    collection = client.get_or_create_collection("neurips_papers")
    all_data = collection.get(include=["embeddings", "metadatas", "documents"])
    paper_embeddings = all_data["embeddings"]     # List of vectors
    paper_metadatas = all_data["metadatas"]       # List of dicts
    # paper_documents = all_data["documents"]     # If needed

    print(f"Fetched {len(paper_embeddings)} papers from DB.")

    # 3. Embed the categories
    model = SentenceTransformer("all-MiniLM-L6-v2")
    category_embeddings = {}
    for cat, desc in category_descriptions.items():
        category_embeddings[cat] = model.encode(desc)

    # 4. For each paper embedding, find best category
    assigned_categories = []
    category_list = list(category_embeddings.keys())  # for iteration
    category_emb_list = list(category_embeddings.values())

    for paper_emb in paper_embeddings:
        # compute similarity with each category
        similarities = cosine_similarity([paper_emb], category_emb_list)[0]  # shape: (num_categories,)
        max_idx = similarities.argmax()
        best_category = category_list[max_idx]
        assigned_categories.append(best_category)

    # 5. Option: store results in a CSV
    output_file = "paper_categories.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "assigned_category", "url"])  # columns
        for meta, cat in zip(paper_metadatas, assigned_categories):
            title = meta["title"]
            url = meta.get("url", "")
            writer.writerow([title, cat, url])

    print(f"Categories assigned. Wrote results to {output_file}.")

    # (Optional) 6. If you want to store categories in DB metadata
    #    you'd do something like:
    #    for i, cat in enumerate(assigned_categories):
    #        # update each document's metadata in the DB
    #        doc_id = paper_metadatas[i]["title"]
    #        # Currently, Chroma doesn't have a direct "update()" method,
    #        # but you can remove + re-add the doc with updated metadata if needed.
    #        # Or store them in a separate "category" collection.

def main():
    categorize_papers()

if __name__ == "__main__":
    main()
