import pandas as pd
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_groq import ChatGroq

load_dotenv()

# ------------------
# Neo4j
# ------------------

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    )
)

# ------------------
# LLM
# ------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# ------------------
# Questions
# ------------------

questions = pd.read_csv(
    "evaluation/questions.csv"
)

results = []

# ------------------
# Benchmark Loop
# ------------------

for i, row in questions.iterrows():

    question = row["question"]

    print(
        f"Processing {i+1}/{len(questions)}"
    )

    # ------------------
    # Entity Extraction
    # ------------------

    entity = llm.invoke(
        f"""
Extract the main entity from the question.

Question:
{question}

Return ONLY the entity name.
"""
    ).content.strip()

    # ------------------
    # Graph Retrieval
    # ------------------

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

    context = "\n".join(graph_facts)

    # ------------------
    # Answer Generation
    # ------------------

    answer = llm.invoke(
        f"""
Answer using ONLY the graph facts.

Facts:
{context}

Question:
{question}
"""
    ).content

    # ------------------
    # Save Result
    # ------------------

    results.append(
        {
            "question": question,
            "answer": answer,
            "contexts": graph_facts,
            "ground_truth": row["ground_truth"]
        }
    )

# ------------------
# Save CSV
# ------------------

pd.DataFrame(results).to_csv(
    "evaluation/graph_rag_results.csv",
    index=False
)

driver.close()

print(
    "\ngraph_rag_results.csv created"
)