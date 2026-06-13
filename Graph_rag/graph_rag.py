import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
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
# LLM
# ------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    # ------------------------
    # Extract entity
    # ------------------------

    entity = llm.invoke(
        f"""
Extract the main AI concept, model,
or entity from the question.

Question:
{question}

Return ONLY the entity name.
"""
    ).content.strip()

    print(f"\nEntity Found: {entity}")

    # ------------------------
    # Query Graph
    # ------------------------

    graph_facts = []

    with driver.session() as session:

        result = session.run(
            """
            MATCH (a)-[r]-(b)
            WHERE toLower(a.name)
            CONTAINS toLower($entity)

            RETURN
                a.name AS source,
                type(r) AS relation,
                b.name AS target

            LIMIT 50
            """,
            entity=entity
        )

        for row in result:

            graph_facts.append(
                f"{row['source']} --{row['relation']}--> {row['target']}"
            )

    print(f"\nRetrieved {len(graph_facts)} graph facts")

    if len(graph_facts) == 0:

        print("\nNo graph facts found.")
        continue

    context = "\n".join(graph_facts)

    print("\nRetrieved Facts:\n")
    print(context[:1500])

    # ------------------------
    # Generate Answer
    # ------------------------

    response = llm.invoke(
        f"""
You are answering questions using
a knowledge graph.

Use ONLY the facts below.

FACTS:
{context}

QUESTION:
{question}

Provide a clear explanation.
Do not invent information that is
not present in the facts.
"""
    )

    print("\nAnswer:\n")
    print(response.content)

driver.close()