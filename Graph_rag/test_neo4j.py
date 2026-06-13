from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    uri,
    auth=(user, password)
)

with driver.session() as session:
    result = session.run(
        "RETURN 'Neo4j Connected!' AS message"
    )

    print(result.single()["message"])

driver.close()