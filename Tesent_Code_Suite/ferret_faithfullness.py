import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ferret import Benchmark
from scipy.stats import rankdata
import torch

# ================================================================
# CONFIGURATION
# ================================================================

# Set your Hugging Face model repo name here
hf_model_name = "PLACE_YOUR_MODEL_NAME"

# CSV test file (expected in current directory)
test_file = "test.csv"

# Batch sizes
prediction_batch_size = 64
ferret_batch_size = 1

# Label mapping
label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ================================================================
# LOAD TEST DATA
# ================================================================

if not os.path.exists(test_file):
    raise FileNotFoundError(f"Test file not found: {test_file}")

df_full = pd.read_csv(test_file)
df_full["final_label"] = df_full["final_label"].astype("category")
df_full["final_label_numeric"] = df_full["final_label"].map(label_map)

texts_all = df_full["Content"].tolist()
labels_all = df_full["final_label"].tolist()
print(f"[INFO] ✅ Loaded test data: {len(df_full)} rows.")

# ================================================================
# PIPELINE FOR SINGLE MODEL (MATCHED ONLY)
# ================================================================

print(f"\n==============================")
print(f"[INFO] 🚀 Starting pipeline for model: {hf_model_name}")
print(f"==============================")

# -----------------------------
# LOAD MODEL & TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    hf_model_name,
    trust_remote_code=True,
    use_safetensors=True
)
model = model.to(device)
model.eval()
print("[INFO] ✅ Model loaded.")

# -----------------------------
# PREDICTIONS
# -----------------------------
predictions = []
print("[INFO] 🔎 Predicting on test set...")

for i in tqdm(range(0, len(texts_all), prediction_batch_size), desc="Predicting"):
    batch_texts = texts_all[i : i + prediction_batch_size]
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
    predictions.extend(preds)

# Store predictions
df = df_full.copy()
df["prediction"] = predictions
df["prediction"] = df["prediction"].astype("int8")

predictions_filename = f"{hf_model_name.replace('/', '_')}_predictions.csv"
df.to_csv(predictions_filename, index=False)
print(f"[INFO] ✅ Predictions saved to {predictions_filename}.")

# -----------------------------
# SPLIT MATCHED ONLY
# -----------------------------
matched_df = df[df["prediction"] == df["final_label_numeric"]].reset_index(drop=True)
print(f"[INFO] ✅ {len(matched_df)} matched rows retained.")

# Save matched for records
matched_df.to_csv(f"{hf_model_name.replace('/', '_')}_matched.csv", index=False)

# -----------------------------
# FERRET ON MATCHED
# -----------------------------
if len(matched_df) > 0:
    ferret_rows = []
    bench = Benchmark(model, tokenizer)
    print(f"[INFO] 🚀 Running FERRET on matched rows...")
    for i in tqdm(range(0, len(matched_df), ferret_batch_size), desc="FERRET (Matched)"):
        batch = matched_df.iloc[i : i + ferret_batch_size]
        for _, row in batch.iterrows():
            text = row["Content"]
            label = int(row["final_label_numeric"])
            try:
                explanations = bench.explain(text, target=label)
                evaluations = bench.evaluate_explanations(explanations, target=label)
            except Exception as ex:
                print(f"[WARN] FERRET failed on matched text: {text}\nReason: {ex}")
                continue
            ferret_row = {
                "Text": text,
                "final_label": row["final_label"],
                "final_label_numeric": label,
                "Annotations": row.get("Annotations", ""),
                "Rationale": row.get("Rationale", ""),
            }
            if explanations:
                ferret_row["Tokens"] = " ".join(explanations[0].tokens)
            for expl, evaluation in zip(explanations, evaluations):
                explainer_name = expl.explainer
                scores = expl.scores
                ranks = rankdata(-np.array(scores), method="min")
                ferret_row[f"{explainer_name}_ImportanceScores"] = " ".join(map(str, scores))
                ferret_row[f"{explainer_name}_RankVector"] = " ".join(map(str, ranks))
                if evaluation and hasattr(evaluation, "evaluation_scores"):
                    for eval_score in evaluation.evaluation_scores:
                        ferret_row[f"{explainer_name}_{eval_score.name}"] = float(eval_score.score)
            ferret_rows.append(ferret_row)
            del explanations
            del evaluations
            gc.collect()
    ferret_filename = f"{hf_model_name.replace('/', '_')}_ferret_matched.csv"
    pd.DataFrame(ferret_rows).to_csv(ferret_filename, index=False)
    print(f"[INFO] ✅ FERRET results saved to {ferret_filename}.")
else:
    print("[INFO] ⚠ No matched rows to run FERRET on.")

print("[INFO] ✅ Pipeline finished for matched rows only!")
