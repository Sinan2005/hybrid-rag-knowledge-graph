import os
import json

from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# Groq
# ------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# ------------------------
# Load PDFs
# ------------------------

documents = []

for file in os.listdir("data"):

    if file.endswith(".pdf"):

        print(f"Loading: {file}")

        loader = PyMuPDFLoader(
            os.path.join("data", file)
        )

        documents.extend(
            loader.load()
        )

print(f"\nLoaded {len(documents)} pages")

# ------------------------
# Split Documents
# ------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    documents
)

print(f"Created {len(chunks)} chunks")

# ------------------------
# Neo4j Helpers
# ------------------------

def create_entity(entity):

    entity = entity.strip().lower()

    with driver.session() as session:

        session.run(
            """
            MERGE (e:Entity {name:$name})
            """,
            name=entity
        )


def create_relationship(
    source,
    relation,
    target
):

    source = source.strip().lower()
    target = target.strip().lower()
    relation = relation.strip().upper()

    with driver.session() as session:

        query = f"""
        MERGE (a:Entity {{name:$source}})
        MERGE (b:Entity {{name:$target}})
        MERGE (a)-[:{relation}]->(b)
        """

        session.run(
            query,
            source=source,
            target=target
        )

# ------------------------
# Extract Graph
# ------------------------

START_CHUNK = 317

for i, chunk in enumerate(
    chunks[START_CHUNK:],
    start=START_CHUNK
):

    text = chunk.page_content

    prompt = f"""
Extract AI entities and relationships.

Return ONLY valid JSON.

Format:

{{
    "entities": [
        "Transformer",
        "Self Attention"
    ],

    "relationships": [
        {{
            "source": "Transformer",
            "target": "Self Attention",
            "relation": "CONTAINS"
        }}
    ]
}}

Allowed relation types:

USES
CONTAINS
BUILT_ON
TRAINED_WITH
IMPROVES
APPLIES_TO

Rules:
- Return valid JSON only.
- No markdown.
- No explanation.
- Relation names must be UPPERCASE.
- Entity names must be concise.

Text:

{text}
"""

    try:

        response = llm.invoke(
            prompt
        )

        content = response.content.strip()

        # Remove markdown if returned

        content = (
            content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        data = json.loads(content)

        entities = data.get(
            "entities",
            []
        )

        relationships = data.get(
            "relationships",
            []
        )

        # ------------------------
        # Store Entities
        # ------------------------

        for entity in entities:

            entity = entity.strip().lower()

            if entity:

                create_entity(
                    entity
                )

        # ------------------------
        # Store Relationships
        # ------------------------

        for rel in relationships:

            source = rel.get(
                "source",
                ""
            ).strip().lower()

            target = rel.get(
                "target",
                ""
            ).strip().lower()

            relation = rel.get(
                "relation",
                ""
            ).strip().upper()

            if source and target and relation:

                create_relationship(
                    source,
                    relation,
                    target
                )

        print(
            f"Chunk {i+1} processed | "
            f"{len(entities)} entities | "
            f"{len(relationships)} relationships"
        )

    except Exception as e:

        print(
            f"Error chunk {i+1}: {e}"
        )

driver.close()

print(
    "\nGraph Created Successfully!"
)