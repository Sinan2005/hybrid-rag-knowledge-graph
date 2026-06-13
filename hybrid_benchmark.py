import os
import pandas as pd

from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# ------------------------
# Neo4j
# ------------------------

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    )
)

# ------------------------
# ChromaDB
# ------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# ------------------------
# LLM
# ------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# ------------------------
# Questions
# ------------------------

questions_df = pd.read_csv(
    "evaluation/questions.csv"
)

results = []

# ------------------------
# Benchmark
# ------------------------

for i, row in questions_df.iterrows():

    question = row["question"]
    ground_truth = row["ground_truth"]

    print(f"Processing {i+1}/{len(questions_df)}")

    # VECTOR RETRIEVAL

    docs = db.similarity_search(
        question,
        k=4
    )

    vector_contexts = [
        doc.page_content
        for doc in docs
    ]

    vector_text = "\n\n".join(
        vector_contexts
    )

    # ENTITY EXTRACTION

    entity = llm.invoke(
        f"""
Extract the main entity.

Question:
{question}

Return ONLY entity name.
"""
    ).content.strip()

    # GRAPH RETRIEVAL

    graph_facts = []

    with driver.session() as session:

        result = session.run(
            """
            MATCH (a)-[r]-(b)

            WHERE toLower(a.name)
            CONTAINS toLower($entity)

            RETURN
                a.name,
                type(r),
                b.name

            LIMIT 50
            """,
            entity=entity
        )

        for r in result:

            graph_facts.append(
                f"{r[0]} {r[1]} {r[2]}"
            )

    graph_text = "\n".join(
        graph_facts
    )

    # HYBRID ANSWER

    answer = llm.invoke(
        f"""
Use BOTH sources.

Vector Context:
{vector_text}

Graph Facts:
{graph_text}

Question:
{question}

Answer:
"""
    ).content

    results.append(
        {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts": str(
                vector_contexts + graph_facts
            )
        }
    )

# ------------------------
# SAVE
# ------------------------

pd.DataFrame(
    results
).to_csv(
    "evaluation/hybrid_rag_results.csv",
    index=False
)

driver.close()

print("\nHybrid benchmark completed!")