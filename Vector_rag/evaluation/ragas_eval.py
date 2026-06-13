import ast
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# -------------------------
# Load Vector RAG Results
# -------------------------

df = pd.read_csv(
    "evaluation/vector_rag_results.csv"
)

# -------------------------
# Convert contexts
# -------------------------

df["contexts"] = df["contexts"].apply(
    ast.literal_eval
)

# -------------------------
# Build Dataset
# -------------------------

dataset = Dataset.from_dict(
    {
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": df["contexts"].tolist(),
        "ground_truth": df["ground_truth"].tolist(),
    }
)

# -------------------------
# Evaluate
# -------------------------

result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

# -------------------------
# Print
# -------------------------

print("\n===== RAGAS RESULTS =====\n")
print(result)

result_df = result.to_pandas()

result_df.to_csv(
    "evaluation/ragas_results.csv",
    index=False
)

print(
    "\nSaved: evaluation/ragas_results.csv"
)