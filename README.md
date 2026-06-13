# Hybrid RAG using Knowledge Graphs and Vector Search

## Overview

This project compares three Retrieval-Augmented Generation (RAG) approaches:

1. Vector RAG
2. Graph RAG
3. Hybrid RAG

The goal is to investigate whether combining vector retrieval with knowledge graph retrieval improves answer quality compared to traditional vector-based retrieval.

The project uses:

- ChromaDB for vector storage
- Neo4j for knowledge graph storage
- Groq Llama 3.3 70B for inference
- HuggingFace Embeddings
- RAGAS for evaluation

---

## Architecture

### Vector RAG

PDF Documents
→ Chunking
→ Embeddings
→ ChromaDB
→ Similarity Search
→ LLM

### Graph RAG

PDF Documents
→ Entity & Relationship Extraction
→ Neo4j Knowledge Graph
→ Graph Retrieval
→ LLM

### Hybrid RAG

PDF Documents
→ ChromaDB Retrieval
+
Neo4j Retrieval
→ Combined Context
→ LLM

---

## Features

- PDF document ingestion
- Automatic chunking
- Vector database retrieval
- Knowledge graph construction
- Entity and relationship extraction
- Hybrid retrieval pipeline
- Benchmarking on custom questions
- RAGAS evaluation

---

## Tech Stack

### LLM

- Groq
- Llama 3.3 70B Versatile

### Vector Database

- ChromaDB

### Knowledge Graph

- Neo4j

### Frameworks

- LangChain
- LangChain Community
- LangChain Groq

### Evaluation

- RAGAS

### Embeddings

- sentence-transformers/all-MiniLM-L6-v2

---

## Project Structure

```text
Hybrid-RAG/
│
├── data/
│   └── PDF files
│
├── chroma_db/
│
├── evaluation/
│   ├── questions.csv
│   ├── hybrid_rag_results.csv
│   ├── hybrid_ragas_eval.py
│   └── hybrid_ragas_results.csv
│
├── graph_builder.py
├── graph_benchmark.py
├── hybrid_benchmark.py
├── vector_benchmark.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/hybrid-rag-knowledge-graph.git
cd hybrid-rag-knowledge-graph
```

Create virtual environment:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Build Knowledge Graph

```bash
python graph_builder.py
```

### Run Vector Benchmark

```bash
python vector_benchmark.py
```

### Run Hybrid Benchmark

```bash
python hybrid_benchmark.py
```

### Evaluate using RAGAS

```bash
python hybrid_ragas_eval.py
```

---

## Evaluation Results

### Vector RAG

| Metric | Score |
|----------|----------|
| Faithfulness | 0.8689 |
| Answer Relevancy | 0.6359 |
| Context Precision | 0.6889 |
| Context Recall | 0.6667 |

### Graph RAG

| Metric | Score |
|----------|----------|
| Faithfulness | 0.6879 |
| Answer Relevancy | 0.2200 |
| Context Precision | 0.0304 |
| Context Recall | 0.2333 |

### Hybrid RAG (Updated)

| Metric | Score |
|----------|----------|
| Faithfulness | 0.7992 |
| Answer Relevancy | 0.7423 |
| Context Precision | 0.6395 |
| Context Recall | 0.7000 |

---

## Key Findings

### Vector RAG

- Highest faithfulness
- Highest context precision
- Strong grounding in source documents

### Hybrid RAG

- Highest answer relevancy
- Highest context recall
- Better understanding of relationships between concepts

### Graph RAG

- Useful for relationship reasoning
- Loses detailed document information when used alone

---

## Conclusion

The results demonstrate that Hybrid RAG combines the strengths of vector retrieval and graph retrieval. While Vector RAG provides highly faithful answers grounded in source documents, Hybrid RAG improves answer relevancy and retrieval coverage by incorporating structured knowledge from a knowledge graph.

---

## Author

**Sinan Tamake**

Machine Learning & Generative AI Enthusiast
