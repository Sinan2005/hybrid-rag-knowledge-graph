# Hybrid RAG: Combining Vector Search and Knowledge Graphs for Enhanced Retrieval

## Project Overview

Retrieval-Augmented Generation (RAG) systems improve Large Language Models (LLMs) by providing external knowledge during inference. Traditional RAG systems rely solely on vector similarity search, which is effective for retrieving semantically relevant document chunks but often struggles to explicitly capture relationships between concepts.

This project explores three different retrieval architectures:

1. **Vector RAG**
2. **Graph RAG**
3. **Hybrid RAG (Vector + Graph)**

The objective is to evaluate whether combining vector retrieval with knowledge graph retrieval can improve answer quality, context coverage, and reasoning capabilities.

The entire system was built using LangChain, Neo4j, ChromaDB, Groq Llama 3.3 70B, and RAGAS.

---

# Motivation

Traditional Vector RAG retrieves information based on semantic similarity.

Example:

Question:

```text
How does Flash Attention improve Transformers?
```

Vector retrieval may return:

```text
Flash Attention reduces memory consumption.

Transformers use self-attention.
```

Although both chunks are relevant, the relationship between them is not explicitly represented.

Knowledge Graphs can store relationships directly:

```text
Transformer IMPROVES Flash Attention
```

This project investigates whether combining both retrieval mechanisms can provide more complete and relevant answers.

---

# System Architecture

## 1. Document Processing Pipeline

PDF Documents
↓
PyMuPDFLoader
↓
Text Extraction
↓
Recursive Character Text Splitting
↓
Chunks

### Chunking Parameters

```python
chunk_size = 1000
chunk_overlap = 200
```

Purpose:

- Preserve context across chunks
- Improve retrieval quality
- Reduce information loss

---

# 2. Vector RAG Pipeline

PDF Chunks
↓
Sentence Transformers Embeddings
↓
ChromaDB
↓
Similarity Search
↓
Retrieved Context
↓
LLM

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Vector Database

```text
ChromaDB
```

### Retrieval Method

```python
similarity_search(question, k=4)
```

Top 4 semantically similar chunks are retrieved.

---

# 3. Knowledge Graph Construction

To build the graph, each chunk is processed by the LLM.

The model extracts:

- Entities
- Relationships

Example Output:

```json
{
  "entities": [
    "Transformer",
    "Self-Attention"
  ],
  "relationships": [
    {
      "source": "Transformer",
      "target": "Self-Attention",
      "relation": "USES"
    }
  ]
}
```

---

# Graph Storage

Neo4j stores:

### Nodes

```text
Transformer
Self-Attention
BERT
GPT
RAG
```

### Relationships

```text
USES
CONTAINS
BUILT_ON
TRAINED_WITH
IMPROVES
APPLIES_TO
```

Example:

```text
Transformer
      |
      USES
      |
Self-Attention
```

---

# Cypher Queries Used

### Create Entity

```cypher
MERGE (e:Entity {name:$name})
```

### Create Relationship

```cypher
MERGE (a:Entity {name:$source})
MERGE (b:Entity {name:$target})
MERGE (a)-[:USES]->(b)
```

### Graph Retrieval

```cypher
MATCH (a)-[r]-(b)

WHERE toLower(a.name)
CONTAINS toLower($entity)

RETURN
    a.name,
    type(r),
    b.name

LIMIT 50
```

---

# 4. Graph RAG Pipeline

Question
↓
Entity Extraction
↓
Neo4j Retrieval
↓
Graph Facts
↓
LLM

Example Retrieved Facts:

```text
Transformer USES Self-Attention

Transformer IMPROVES Flash Attention

BERT BUILT_ON Transformer
```

---

# 5. Hybrid RAG Pipeline

Question
↓
Vector Retrieval
↓
Graph Retrieval
↓
Combined Context
↓
LLM

Context supplied to the LLM:

```text
Vector Context:
...

Graph Facts:
...
```

The model receives both detailed document information and explicit relationships.

---

# Technologies Used

## LLM

- Groq
- Llama 3.3 70B Versatile

## Frameworks

- LangChain
- LangChain Community
- LangChain Groq

## Vector Search

- ChromaDB
- Sentence Transformers

## Knowledge Graph

- Neo4j

## Evaluation

- RAGAS

## Data Processing

- Pandas
- PyMuPDF

---

# Evaluation Methodology

The systems were evaluated using RAGAS on a custom benchmark dataset.

Metrics used:

### Faithfulness

Measures whether the generated answer is supported by the retrieved context.

### Answer Relevancy

Measures how well the answer addresses the user's question.

### Context Precision

Measures how much of the retrieved context is actually useful.

### Context Recall

Measures how much relevant information was successfully retrieved.

---

# Results

## Vector RAG

| Metric | Score |
|----------|----------|
| Faithfulness | 0.8689 |
| Answer Relevancy | 0.6359 |
| Context Precision | 0.6889 |
| Context Recall | 0.6667 |

---

## Graph RAG

| Metric | Score |
|----------|----------|
| Faithfulness | 0.6879 |
| Answer Relevancy | 0.2200 |
| Context Precision | 0.0304 |
| Context Recall | 0.2333 |

---

## Hybrid RAG (Final Version)

| Metric | Score |
|----------|----------|
| Faithfulness | 0.7992 |
| Answer Relevancy | 0.7423 |
| Context Precision | 0.6395 |
| Context Recall | 0.7000 |

---

# Analysis

## Why Vector RAG Achieved Higher Faithfulness

Vector RAG retrieves original document chunks directly from the source PDFs.

The LLM mostly summarizes retrieved information rather than inferring new facts.

This leads to highly grounded answers.

---

## Why Hybrid RAG Achieved Higher Answer Relevancy

Hybrid RAG combines:

- Detailed textual context
- Explicit graph relationships

The graph provides structured knowledge that helps the model better understand how concepts are connected.

This results in answers that are more focused on the user's question.

---

## Why Hybrid RAG Improved Context Recall

Information is retrieved from two independent sources:

1. ChromaDB
2. Neo4j

Even if vector search misses some information, graph retrieval can provide additional relevant facts.

This improves overall retrieval coverage.

---

# Key Learnings

Through this project I learned:

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- ChromaDB
- Knowledge Graph Construction
- Neo4j
- Cypher Query Language
- LangChain
- Hybrid Retrieval Architectures
- Entity and Relationship Extraction
- LLM Evaluation using RAGAS
- Retrieval Metrics Analysis
- Prompt Engineering

---

# Future Improvements

- Graph-aware retrievers
- Entity linking and disambiguation
- Hybrid ranking strategies
- Larger benchmark datasets
- Multi-hop graph reasoning
- Agentic RAG using LangGraph
- Graph embeddings

---

# Author

## Sinan Tamake

Machine Learning | Deep Learning | Generative AI

Project Focus:
- RAG Systems
- Knowledge Graphs
- LLM Evaluation
- Agentic AI
