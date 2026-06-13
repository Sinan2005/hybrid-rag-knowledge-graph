from dotenv import load_dotenv
load_dotenv()

import ast
import pandas as pd

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# ------------------
# Load Graph Results
# ------------------

df = pd.read_csv(
    "evaluation/graph_rag_results.csv"
)

# ------------------
# Convert contexts
# ------------------

df["contexts"] = df["contexts"].apply(
    ast.literal_eval
)

# ------------------
# Dataset
# ------------------

dataset = Dataset.from_dict(
    {
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": df["contexts"].tolist(),
        "ground_truth": df["ground_truth"].tolist(),
    }
)

# ------------------
# Evaluate
# ------------------

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print("\n===== GRAPH RAG RESULTS =====\n")
print(result)

result.to_pandas().to_csv(
    "evaluation/graph_ragas_results.csv",
    index=False
)

print(
    "\nSaved: evaluation/graph_ragas_results.csv"
)