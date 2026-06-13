import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
# ---------------------
# Load Questions
# ---------------------

questions_df = pd.read_csv(
    "evaluation/questions.csv"
)

# ---------------------
# Load Vector DB
# ---------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# ---------------------
# Load LLM
# ---------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

results = []

# ---------------------
# Evaluation Loop
# ---------------------

for _, row in questions_df.iterrows():

    question = row["question"]

    ground_truth = row["ground_truth"]

    docs = db.similarity_search(
        question,
        k=4
    )

    contexts = [
        doc.page_content
        for doc in docs
    ]

    context_text = "\n\n".join(
        contexts
    )

    prompt = f"""
Answer ONLY using the context.

Context:
{context_text}

Question:
{question}
"""

    response = llm.invoke(
        prompt
    )

    answer = response.content

    results.append({
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "contexts": contexts
    })

    print(
        f"Completed: {question}"
    )

# ---------------------
# Save Results
# ---------------------

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "evaluation/vector_rag_results.csv",
    index=False
)

print(
    "\nEvaluation data generated successfully!"
)