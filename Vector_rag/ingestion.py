import os
import shutil

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------
# Configuration
# -----------------------

DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"

# -----------------------
# Remove old vector DB
# -----------------------

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
    print("Old ChromaDB removed")

# -----------------------
# Load PDFs
# -----------------------

documents = []

for file in os.listdir(DATA_FOLDER):

    if file.endswith(".pdf"):

        file_path = os.path.join(
            DATA_FOLDER,
            file
        )

        print(f"Loading: {file}")

        loader = PyMuPDFLoader(file_path)

        docs = loader.load()

        # Add metadata
        for doc in docs:

            doc.metadata["filename"] = file

            if "source" not in doc.metadata:
                doc.metadata["source"] = file

        documents.extend(docs)

print(f"\nLoaded {len(documents)} pages")

# -----------------------
# Chunking
# -----------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    documents
)

print(f"Created {len(chunks)} chunks")

# -----------------------
# Embeddings
# -----------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings...")

# -----------------------
# Vector Store
# -----------------------

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)

print("\nVector database created successfully!")

# -----------------------
# Statistics
# -----------------------

unique_files = set(
    doc.metadata["filename"]
    for doc in documents
)

print("\n========== SUMMARY ==========")
print(f"PDF Files Loaded : {len(unique_files)}")
print(f"Pages Loaded     : {len(documents)}")
print(f"Chunks Created   : {len(chunks)}")
print("=============================")