# Hybrid RAG: Enhancing Retrieval-Augmented Generation with Knowledge Graphs and Vector Search

## Overview

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, despite their impressive performance, they suffer from several limitations, including hallucinations, outdated knowledge, and the inability to access domain-specific information that was not present during training.

Retrieval-Augmented Generation (RAG) addresses these limitations by retrieving relevant information from external knowledge sources and providing it to the LLM during inference. Traditional RAG systems rely primarily on vector similarity search, which retrieves semantically similar document chunks using embeddings.

While vector retrieval is highly effective for finding relevant information, it often lacks an explicit understanding of relationships between concepts. Knowledge Graphs provide a complementary approach by representing information as interconnected entities and relationships, enabling structured retrieval and relationship-aware reasoning.

This project investigates whether combining vector retrieval with knowledge graph retrieval can improve answer quality, retrieval coverage, and contextual understanding.

The project implements and compares three retrieval architectures:

* Vector RAG
* Graph RAG
* Hybrid RAG (Vector + Graph)

The systems are evaluated using the RAGAS evaluation framework across multiple retrieval and generation metrics.

---

# Motivation

Modern LLMs generate responses based on patterns learned during training. As a result, they may:

* Produce hallucinated information
* Lack access to recently published information
* Struggle with domain-specific knowledge
* Fail to reason over explicit relationships between concepts

For example, consider the question:

```text
How does Flash Attention improve Transformers?
```

A traditional Vector RAG system may retrieve:

```text
Flash Attention reduces memory consumption.

Transformers use self-attention mechanisms.
```

Although both chunks are relevant, the relationship between them is not explicitly represented.

A Knowledge Graph can represent this relationship directly:

```text
Transformer
     |
 IMPROVES
     |
Flash Attention
```

By combining both retrieval methods, the system can leverage:

* Detailed textual information from vector search
* Explicit relationships from graph search

This forms the foundation of the Hybrid RAG architecture developed in this project.

---

# Project Objectives

The primary objectives of this project are:

### 1. Build a Traditional Vector RAG System

Create a retrieval pipeline using:

* PDF documents
* Sentence embeddings
* ChromaDB vector storage
* Semantic similarity search

### 2. Construct a Knowledge Graph

Transform unstructured PDF documents into a structured graph representation using:

* Entity extraction
* Relationship extraction
* Neo4j graph storage

### 3. Implement Graph-Based Retrieval

Retrieve graph facts relevant to a user query and provide them to the LLM.

### 4. Build a Hybrid Retrieval Architecture

Combine:

* Vector retrieval
* Graph retrieval

into a unified context generation pipeline.

### 5. Evaluate System Performance

Use RAGAS metrics to compare:

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall

across all retrieval architectures.

---

# Background

## What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation is a technique that combines external retrieval systems with Large Language Models.

Instead of relying solely on the information stored within model weights, RAG systems retrieve relevant information at runtime.

### Traditional LLM Workflow

```text
User Question
      ↓
     LLM
      ↓
    Answer
```

The model answers purely from its training data.

---

### RAG Workflow

```text
User Question
      ↓
  Retriever
      ↓
Relevant Context
      ↓
     LLM
      ↓
    Answer
```

The model now has access to external knowledge.

This improves:

* Accuracy
* Explainability
* Domain adaptation
* Knowledge freshness

---

# Vector RAG Architecture

The Vector RAG system retrieves relevant information using semantic similarity.

## Step 1: Document Loading

The project uses PyMuPDFLoader to load PDF documents.

```python
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()
```

Each PDF page is converted into a LangChain document object.

---

## Step 2: Text Chunking

Large documents cannot be embedded directly.

Therefore, documents are divided into smaller chunks.

The project uses:

```python
RecursiveCharacterTextSplitter
```

Configuration:

```python
chunk_size = 1000
chunk_overlap = 200
```

### Why Chunking?

Chunking improves retrieval performance because:

* Smaller chunks are easier to embed
* Retrieval becomes more focused
* Important information is less likely to be diluted

The overlap ensures that information spanning multiple chunks is preserved.

---

## Step 3: Embedding Generation

Each chunk is converted into a dense vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings capture semantic meaning rather than exact words.

For example:

```text
Transformer Architecture

Self-Attention Mechanism
```

will be placed close together in vector space.

---

## Step 4: Vector Storage

The generated embeddings are stored in ChromaDB.

ChromaDB acts as the project's vector database.

Responsibilities:

* Store embeddings
* Index vectors
* Perform similarity search
* Return relevant chunks

---

## Step 5: Retrieval

When a question is received:

```python
docs = db.similarity_search(
    question,
    k=4
)
```

The system retrieves the top 4 semantically similar chunks.

These chunks become the context supplied to the LLM.

---

# Knowledge Graph Architecture

Unlike Vector RAG, Graph RAG focuses on relationships.

Knowledge Graphs represent information as:

* Nodes (Entities)
* Edges (Relationships)

This structure enables retrieval of connected concepts that may not appear together in the original document.

## Why Knowledge Graphs?

Consider the following statements:

```text
Transformers use Self-Attention.

BERT is built on Transformers.
```

A Knowledge Graph stores:

Transformer
|
USES
|
Self-Attention

BERT
|
BUILT_ON
|
Transformer

The graph explicitly captures relationships that may otherwise be hidden within text.
# Entity and Relationship Extraction

A key component of this project is converting unstructured document text into a structured knowledge graph.

Unlike traditional databases that store information in tables, knowledge graphs represent information as interconnected entities and relationships.

To build this graph automatically, an LLM-based extraction pipeline was developed.

---

## Entity Extraction

For every chunk generated during preprocessing, the chunk text is sent to the LLM.

The model identifies important AI-related concepts including:

* Models
* Architectures
* Algorithms
* Techniques
* Frameworks
* Datasets

Example chunk:

```text
Transformers use self-attention mechanisms to capture long-range dependencies in text.
```

The LLM extracts:

```json
{
    "entities": [
        "Transformer",
        "Self-Attention"
    ]
}
```

These entities become nodes inside Neo4j.

---

## Relationship Extraction

Entity extraction alone is insufficient because entities without relationships provide very little value.

The system also extracts relationships between entities.

Example:

```text
Transformers use self-attention mechanisms.
```

The extracted relationship becomes:

```json
{
    "source": "Transformer",
    "target": "Self-Attention",
    "relation": "USES"
}
```

This relationship becomes an edge inside the graph.

---

## Relationship Types

To maintain consistency and avoid noisy graph construction, only a predefined set of relationship types was allowed.

### USES

Represents dependency relationships.

Example:

```text
Transformer USES Self-Attention
```

---

### CONTAINS

Represents component relationships.

Example:

```text
BERT CONTAINS Encoder Layers
```

---

### BUILT_ON

Represents foundational relationships.

Example:

```text
BERT BUILT_ON Transformer
```

---

### TRAINED_WITH

Represents training methodology relationships.

Example:

```text
BERT TRAINED_WITH Masked Language Modeling
```

---

### IMPROVES

Represents optimization or enhancement relationships.

Example:

```text
Flash Attention IMPROVES Transformer
```

---

### APPLIES_TO

Represents application relationships.

Example:

```text
Transfer Learning APPLIES_TO NLP
```

---

# Neo4j Knowledge Graph Construction

After extraction, entities and relationships are stored inside Neo4j.

Neo4j is a graph database that stores information as:

* Nodes
* Relationships
* Properties

---

## Node Structure

Each extracted entity becomes a node.

Example:

```cypher
(:Entity {
    name: "Transformer"
})
```

---

## Relationship Structure

Relationships connect two nodes.

Example:

```cypher
(:Entity {name:"Transformer"})
      -[:USES]->
(:Entity {name:"Self-Attention"})
```

---

# Cypher Queries Used

Cypher is Neo4j's query language.

It serves a similar purpose to SQL but is specifically designed for graph databases.

---

## Creating Nodes

```cypher
MERGE (e:Entity {name:$name})
```

### Explanation

MERGE:

* Searches for an existing node
* Creates one if it does not exist

This prevents duplicate entities.

For example:

```text
Transformer
Transformer
Transformer
```

will only create one node.

---

## Creating Relationships

```cypher
MERGE (a:Entity {name:$source})
MERGE (b:Entity {name:$target})
MERGE (a)-[:USES]->(b)
```

### Explanation

First:

```cypher
MERGE (a:Entity {name:$source})
```

creates or retrieves the source node.

---

Second:

```cypher
MERGE (b:Entity {name:$target})
```

creates or retrieves the target node.

---

Finally:

```cypher
MERGE (a)-[:USES]->(b)
```

creates the relationship.

If the relationship already exists, Neo4j does not duplicate it.

---

# Graph Retrieval Pipeline

After graph construction, the system can retrieve graph facts relevant to a user query.

The retrieval process consists of three stages.

---

## Stage 1: Entity Identification

Given a question:

```text
How does Flash Attention improve Transformers?
```

The LLM first identifies the main entity.

Example output:

```text
Flash Attention
```

This entity becomes the retrieval key.

---

## Stage 2: Neo4j Query

The following Cypher query is executed:

```cypher
MATCH (a)-[r]-(b)

WHERE toLower(a.name)
CONTAINS toLower($entity)

RETURN
    a.name AS source,
    type(r) AS relation,
    b.name AS target

LIMIT 50
```

---

## Query Breakdown

### MATCH

```cypher
MATCH (a)-[r]-(b)
```

Finds connected nodes and relationships.

---

### WHERE

```cypher
WHERE toLower(a.name)
CONTAINS toLower($entity)
```

Performs case-insensitive matching.

Example:

```text
Transformer
transformer
TRANSFORMER
```

All produce the same result.

---

### RETURN

```cypher
RETURN
a.name,
type(r),
b.name
```

Returns:

```text
Transformer USES Self-Attention
```

instead of raw Neo4j objects.

---

### LIMIT

```cypher
LIMIT 50
```

Prevents excessively large graph retrievals.

---

# Graph Context Generation

Retrieved graph facts are converted into text.

Example:

```text
Transformer USES Self-Attention

BERT BUILT_ON Transformer

Flash Attention IMPROVES Transformer
```

This becomes the graph context provided to the LLM.

Unlike document chunks, graph facts are highly structured and relationship-focused.

This allows the model to understand how concepts are connected rather than relying solely on semantic similarity.
# Hybrid RAG Architecture

The Hybrid RAG system is the primary contribution of this project.

It combines the strengths of:

* Vector Retrieval (ChromaDB)
* Knowledge Graph Retrieval (Neo4j)

to generate answers using both detailed textual information and explicit relationships between concepts.

---

## Why Hybrid RAG?

Both retrieval methods have strengths and weaknesses.

### Vector RAG Strengths

* Retrieves rich textual information
* Preserves document details
* Produces highly faithful answers

### Vector RAG Limitations

* Relationships are implicit
* Multi-hop reasoning is difficult
* Important connections may be hidden across multiple chunks

---

### Graph RAG Strengths

* Explicit relationships
* Structured retrieval
* Better conceptual understanding

### Graph RAG Limitations

* Loses detailed textual information
* Relationships alone are insufficient for many questions

---

### Hybrid RAG Solution

Hybrid RAG combines both sources.

The system retrieves:

```text
Document Chunks
+
Graph Facts
```

and supplies both to the LLM.

This allows the model to benefit from:

* Rich context
* Structured relationships
* Better retrieval coverage

---

# Hybrid Retrieval Workflow

## Step 1: User Question

Example:

```text
How does Flash Attention improve Transformers?
```

---

## Step 2: Vector Retrieval

The system retrieves:

```text
Flash Attention reduces memory consumption.

Flash Attention reduces attention complexity.
```

---

## Step 3: Graph Retrieval

The system retrieves:

```text
Flash Attention IMPROVES Transformer

Transformer USES Self-Attention
```

---

## Step 4: Context Combination

The retrieved contexts are merged.

```text
Vector Context:

Flash Attention reduces memory consumption.

Flash Attention reduces attention complexity.

Graph Facts:

Flash Attention IMPROVES Transformer

Transformer USES Self-Attention
```

---

## Step 5: LLM Generation

The combined context is provided to Groq Llama 3.3.

The model generates the final answer using both retrieval sources.

---

# Benchmarking Pipeline

After implementing all retrieval systems, a benchmarking framework was developed.

The objective was to compare:

* Vector RAG
* Graph RAG
* Hybrid RAG

under identical conditions.

---

## Evaluation Dataset

A custom evaluation dataset was created.

The dataset contains:

* User questions
* Ground-truth answers

Example:

| Question                   | Ground Truth         |
| -------------------------- | -------------------- |
| What is Flash Attention?   | Detailed explanation |
| What is Transfer Learning? | Detailed explanation |

The benchmark contains 30 evaluation questions.

---

## Answer Generation

For every question:

1. Retrieve context
2. Generate answer
3. Save output

Example structure:

```python
results.append(
{
    "question": question,
    "answer": answer,
    "ground_truth": ground_truth,
    "contexts": contexts
}
)
```

The generated results are saved to:

```text
evaluation/hybrid_rag_results.csv
```

---

# RAGAS Evaluation Framework

To objectively evaluate performance, the project uses RAGAS.

RAGAS is specifically designed to evaluate Retrieval-Augmented Generation systems.

Unlike traditional NLP metrics, RAGAS evaluates both:

* Retrieval Quality
* Generation Quality

---

## Why RAGAS?

Traditional metrics such as:

* BLEU
* ROUGE
* Accuracy

are often insufficient for evaluating RAG systems.

RAGAS evaluates:

* Grounding
* Context Quality
* Retrieval Coverage

which are critical for RAG architectures.

---

# Evaluation Metrics

## Faithfulness

### Definition

Measures how well the generated answer is supported by the retrieved context.

### Purpose

Detect hallucinations.

### High Score Means

* Answer is grounded in context
* Minimal unsupported claims

---

### Example

Retrieved Context:

```text
Flash Attention reduces memory consumption.
```

Answer:

```text
Flash Attention reduces memory consumption.
```

High faithfulness.

---

Answer:

```text
Flash Attention reduces memory consumption and doubles training speed.
```

If "doubles training speed" is not present in the context:

Faithfulness decreases.

---

# Answer Relevancy

### Definition

Measures how well the answer addresses the user's question.

### Purpose

Evaluate usefulness of generated responses.

---

Question:

```text
How does Flash Attention improve Transformers?
```

Answer:

```text
Flash Attention reduces memory requirements in Transformers.
```

High relevancy.

---

Answer:

```text
Transformers are deep learning architectures.
```

Low relevancy.

---

# Context Precision

### Definition

Measures how much of the retrieved context is actually useful.

### Purpose

Detect retrieval noise.

---

High Precision:

```text
Question:
Transformer

Retrieved:
Transformer chunk
Transformer chunk
Transformer chunk
```

---

Low Precision:

```text
Question:
Transformer

Retrieved:
Transformer chunk
Weather chunk
Database chunk
```

Much of the retrieved context is irrelevant.

---

# Context Recall

### Definition

Measures how much relevant information was successfully retrieved.

### Purpose

Evaluate retrieval completeness.

---

High Recall:

Most information required for answering the question is retrieved.

---

Low Recall:

Important information is missing from retrieved context.

---

# Experimental Results

## Vector RAG

| Metric            | Score  |
| ----------------- | ------ |
| Faithfulness      | 0.8689 |
| Answer Relevancy  | 0.6359 |
| Context Precision | 0.6889 |
| Context Recall    | 0.6667 |

---

## Hybrid RAG

| Metric            | Score  |
| ----------------- | ------ |
| Faithfulness      | 0.7992 |
| Answer Relevancy  | 0.7423 |
| Context Precision | 0.6395 |
| Context Recall    | 0.7000 |

---

# Results Analysis

The evaluation results reveal important trade-offs between retrieval architectures.

---

## Why Vector RAG Achieved Higher Faithfulness

Vector RAG retrieves actual document chunks.

The generated answer is usually a direct summary of retrieved text.

Because the model performs less inference, the answer remains highly grounded.

Result:

```text
Faithfulness = 0.8689
```

---

## Why Hybrid RAG Achieved Higher Answer Relevancy

The graph introduces explicit relationships.

Instead of relying solely on semantic similarity, the model receives structured knowledge.

This improves understanding of how concepts connect.

Result:

```text
Answer Relevancy = 0.7423
```

which exceeds Vector RAG.

---

## Why Hybrid RAG Achieved Higher Context Recall

Hybrid retrieval uses two independent knowledge sources:

1. ChromaDB
2. Neo4j

Even if vector retrieval misses information, graph retrieval may recover related facts.

Result:

```text
Context Recall = 0.7000
```

which exceeds Vector RAG.
# Challenges Faced During Development

Building a Hybrid RAG system required solving several practical engineering challenges. While the final architecture appears straightforward, significant effort was required to ensure stable graph construction, retrieval quality, and evaluation reliability.

---

## 1. Entity Extraction Consistency

One of the first challenges was maintaining consistency in entity names.

Consider the following outputs:

```text
Transformer
transformer
Transformers
```

Although they refer to the same concept, Neo4j treats them as different nodes.

This resulted in:

```text
Transformer

transformer

Transformers
```

being stored as three separate entities.

### Solution

All entities were normalized before insertion:

```python
entity = entity.strip().lower()
```

This reduced duplication and improved graph quality.

---

## 2. LLM JSON Formatting Errors

The graph construction pipeline relied on the LLM returning valid JSON.

Expected output:

```json
{
  "entities": [
    "Transformer",
    "Self-Attention"
  ]
}
```

However, the model occasionally returned:

````text
```json
{
 ...
}
```
````

or malformed JSON.

### Solution

Additional preprocessing was introduced:

````python
content = content.replace(
    "```json",
    ""
).replace(
    "```",
    ""
)
````

before parsing.

---

## 3. Graph Noise

Not every extracted relationship was useful.

Example:

```text
Transformer CONTAINS Model
```

While technically correct, such relationships contributed little to retrieval quality.

### Impact

Excessive graph noise reduced:

* Context Precision
* Faithfulness

during evaluation.

### Mitigation

A restricted set of relationship types was enforced:

* USES
* CONTAINS
* BUILT_ON
* TRAINED_WITH
* IMPROVES
* APPLIES_TO

This significantly improved graph quality.

---

## 4. API Rate Limits

Knowledge graph construction required processing hundreds of chunks.

Each chunk involved:

1. Sending text to the LLM
2. Extracting entities
3. Extracting relationships

This consumed a large number of API tokens.

### Issue

Groq token limits were occasionally exceeded during:

* Graph construction
* Benchmark generation
* RAGAS evaluation

### Solution

Processing was resumed using:

```python
START_CHUNK = ...
```

allowing interrupted runs to continue from the last processed chunk.

---

## 5. Hybrid Retrieval Balancing

A major challenge was balancing:

* Graph retrieval
* Vector retrieval

Too many graph facts reduced precision.

Too few graph facts reduced recall.

Finding an appropriate retrieval strategy required multiple experiments.

---

# Lessons Learned

This project provided practical experience in building end-to-end Generative AI systems.

Key learnings include:

---

## Retrieval Is More Important Than the LLM

Initially, the focus was on selecting the best model.

However, experiments showed that retrieval quality often has a larger impact than model choice.

Poor retrieval results in poor answers regardless of model capability.

---

## Knowledge Graphs Are Powerful but Incomplete

Graphs excel at representing relationships.

Example:

```text
Transformer USES Self-Attention

BERT BUILT_ON Transformer
```

However, relationships alone do not provide sufficient detail for answering many questions.

Detailed textual context remains essential.

---

## Evaluation Matters

Without evaluation, it is difficult to determine whether architectural changes actually improve performance.

RAGAS provided objective measurements for:

* Faithfulness
* Relevancy
* Precision
* Recall

allowing meaningful comparisons between retrieval systems.

---

## Hybrid Systems Involve Trade-Offs

The results demonstrated that improvements in one metric can reduce another.

For example:

* Higher recall may reduce precision
* More reasoning may reduce faithfulness

Understanding these trade-offs is critical when designing production RAG systems.

---

# Future Improvements

Several enhancements can further improve the system.

---

## Multi-Hop Graph Retrieval

Current retrieval uses:

```text
Entity
↓
Immediate Relationships
```

Future versions can perform:

```text
Entity
↓
Related Entity
↓
Related Entity
```

enabling deeper reasoning.

---

## Graph Embeddings

The current graph relies on symbolic retrieval.

Future work could incorporate:

* Node embeddings
* Graph embeddings

to improve retrieval quality.

---

## Re-Ranking

Retrieved contexts can be re-ranked before generation.

Possible approaches:

* Cross-Encoder Re-ranking
* LLM-based Re-ranking
* Hybrid Scoring

This may improve precision.

---

## Agentic RAG with LangGraph

The current architecture follows a fixed retrieval workflow.

Future versions can use LangGraph to create an agent capable of:

* Deciding which retriever to use
* Querying multiple tools
* Planning retrieval steps
* Performing iterative reasoning

Example:

```text
Question
↓
Agent
↓
Vector Search
↓
Graph Search
↓
Reasoning
↓
Answer
```

---

## Multi-Agent Architectures

Potential agents:

### Retrieval Agent

Responsible for context collection.

### Graph Agent

Responsible for graph exploration.

### Evaluation Agent

Responsible for validating generated answers.

---

## Larger Evaluation Dataset

Current benchmark:

```text
30 Questions
```

Future versions could include:

```text
100+
Questions
```

for more reliable evaluation.

---

# Repository Structure

```text
Hybrid-RAG/
│
├── data/
│   └── PDF documents
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
├── vector_benchmark.py
├── graph_benchmark.py
├── hybrid_benchmark.py
│
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

---

# Conclusion

This project explored the integration of vector retrieval and knowledge graph retrieval within a Retrieval-Augmented Generation framework.

Three retrieval architectures were implemented and evaluated:

* Vector RAG
* Graph RAG
* Hybrid RAG

The results demonstrate that:

* Vector RAG provides highly faithful and precise answers.
* Knowledge Graphs provide structured relationships between concepts.
* Hybrid RAG combines the strengths of both approaches.

Experimental evaluation showed that Hybrid RAG achieved higher answer relevancy and context recall by leveraging both semantic similarity and explicit graph relationships.

The project highlights the importance of retrieval quality, structured knowledge representation, and systematic evaluation when building modern AI systems.

Beyond implementation, this work provided hands-on experience with:

* LangChain
* ChromaDB
* Neo4j
* Knowledge Graph Construction
* Retrieval-Augmented Generation
* Prompt Engineering
* LLM Evaluation
* RAGAS
* Generative AI System Design

and serves as a foundation for future work in Agentic AI, LangGraph, and advanced retrieval architectures.

---

## Author

**Sinan Tamake**

B.Tech Computer Science Engineering

Areas of Interest:

* Generative AI
* Retrieval-Augmented Generation
* Knowledge Graphs
* Agentic AI
* Machine Learning
* Deep Learning
* MLOps
* Large Language Models
