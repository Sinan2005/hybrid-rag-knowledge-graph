from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    )
)

query_entity = input(
    "Enter entity: "
)

with driver.session() as session:

    result = session.run(
        """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE toLower(a.name)
        CONTAINS toLower($entity)

        RETURN a.name,
               type(r),
               b.name
        LIMIT 20
        """,
        entity=query_entity
    )

    print("\nKnowledge Graph Results\n")

    for row in result:

        print(
            f"{row[0]} "
            f"--{row[1]}--> "
            f"{row[2]}"
        )

driver.close()