import os
import ast
import pandas as pd

from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv()

# --------------------------
# Load Hybrid Results
# --------------------------

df = pd.read_csv(
    "evaluation/hybrid_rag_results.csv"
)

# --------------------------
# Convert contexts
# --------------------------

contexts = []

for c in df["contexts"]:

    try:
        contexts.append(
            ast.literal_eval(c)
        )
    except:
        contexts.append(
            [str(c)]
        )

# --------------------------
# Create Dataset
# --------------------------

dataset = Dataset.from_dict(
    {
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": contexts,
        "ground_truth": df["ground_truth"].tolist(),
    }
)

# --------------------------
# Evaluate
# --------------------------

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
)

# --------------------------
# Print Results
# --------------------------

print("\n===== HYBRID RAG RESULTS =====\n")
print(result)

# --------------------------
# Save Results
# --------------------------

pd.DataFrame(
    [result]
).to_csv(
    "evaluation/hybrid_ragas_results.csv",
    index=False
)

print(
    "\nSaved: evaluation/hybrid_ragas_results.csv"
)