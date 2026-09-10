# AI-Powered Document Parsing & Spatial Knowledge Graph Architecture

A simple end-to-end document intelligence pipeline designed to ingest unstructured PDFs, extract deep structural
layouts, and perform NLP and representation learning tasks.

> **Note**: This project is currently a work in progress.

## Project Overview

The goal of this project is to identify research gaps through an automated NLP process only requiring the source
document and anchor terms that the end-user wishes to focus on, This project is mostly for educational purposes as a
more modern recreation of older work I have produced.

## Key Features and  Architecture

* **Structural Layout Extraction:** Ingests PDFs via "PyMuPDF" and leverages Hugging Face Transformers to run a
["PP-DocLayoutV3"](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) model. This extracts reading order,
precise bounding boxes, and structural labels.
* **Schema Enforcement:** Programmatically reconstructs unstructured bounding boxes into clean, deterministic Markdown
and strict "JSON" schemas to map internal metadata by section and paper.
* **Thematic Information Retrieval (IR):** Deploys ["SciBERT"](https://huggingface.co/allenai/scibert_scivocab_uncased)
for deep domain-specific contextual embeddings to automatically discover and extend anchor-word phrases.
* **Automated Clustering:** Integrates ["BERTopic"](https://huggingface.co/MaartenGr/BERTopic_ArXiv) to identify latent
themes and automate document clustering across the ingested corpus.
* **Representation Learning:** Constructs a custom network graph pipeline to map and discover missing intersections
between distinct research domains, automating overall content enrichment.

## Roadmap

- [10%] PDF ingestion wrapper with PyMuPDF 
- [0%] Finalize PP-DocLayoutV3 transformer integration
- [0%] Complete rigid JSON schema validation
- [0%] Implement SciBERT expansion and BERTopic clustering models
- [0%] Build network graph representation layer