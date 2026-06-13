import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper RAG")

# -----------------------
# Embeddings
# -----------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------
# Vector Store
# -----------------------

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# -----------------------
# LLM
# -----------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

question = st.text_input(
    "Ask a question about your research papers"
)

if question:

    docs = retriever.invoke(question)

    st.info(f"Retrieved {len(docs)} chunks")

    context = ""

    for i, doc in enumerate(docs, 1):
        context += f"\n\n[Document {i}]\n"
        context += doc.page_content

    prompt = f"""
You are an AI research assistant.

Rules:
1. Answer ONLY from the provided context.
2. If the answer is not present, say:
   "I could not find that information in the retrieved documents."
3. Do not make up facts.
4. Keep answers concise and accurate.

Context:
{context}

Question:
{question}

Answer:
"""

    with st.spinner("Generating answer..."):
        response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)

    # -----------------------
    # Sources
    # -----------------------

    st.subheader("Sources")

    for i, doc in enumerate(docs, 1):

        source = doc.metadata.get(
            "source",
            "Unknown Source"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        st.write(
            f"📄 Document {i}: {source} | Page {page}"
        )

    # -----------------------
    # Retrieved Chunks
    # -----------------------

    with st.expander("Retrieved Context"):

        for i, doc in enumerate(docs, 1):

            source = doc.metadata.get(
                "source",
                "Unknown Source"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            st.markdown(
                f"### Chunk {i}"
            )

            st.caption(
                f"Source: {source} | Page: {page}"
            )

            st.write(
                doc.page_content[:1500]
            )