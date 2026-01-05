import os
import ferret
from ferret import Benchmark
import csv
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ==============================
# File path and Data Loading
# ==============================
input_file = "modelname_ferret_matched.csv"  # Columns: 'Text', 'final_label', 'final_label_numeric', 'Annotations', 'Rationale', plus FERRET columns

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

df = pd.read_csv(input_file)

# ==============================
# Model and Plausibility Configs
# ==============================
hf_model_names = [
    #keep your model names as "model_name"
]
label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
inv_label_map = {v: k for k, v in label_map.items()}
max_length = 128

# ==============================
# Rationales to Attention Vectors
# ==============================
def generate_attention_vectors_from_rationales(df, tokenizer, max_length=128):
    """
    For each row, generate attention vectors for each annotator.
    Vector is zero if annotator's label does not match final label.
    """
    all_attention_vectors = []
    token_lengths = []
    max_annotators = 0

    for _, row in df.iterrows():
        text = str(row["Text"])
        final_label_id = row["final_label_numeric"]
        final_label = inv_label_map[final_label_id]

        encoding = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            max_length=max_length,
            truncation=True
        )
        offsets = encoding["offset_mapping"]
        real_token_indices = [i for i, (start, end) in enumerate(offsets) if start != end and start >= 0]
        num_real_tokens = len(real_token_indices)
        token_lengths.append(num_real_tokens)

        annotations = str(row["Annotations"]).split("|")
        rationales = str(row["Rationale"]).split("|")
        max_annotators = max(max_annotators, len(annotations))

        row_attention_vectors = []

        for annot_label, annot_rationale in zip(annotations, rationales):
            vec = np.zeros(num_real_tokens, dtype=np.float32)
            # Set all zeros and skip if annotator label does not match final label
            if not annot_label.strip() or annot_label.split("-")[0].strip() != final_label:
                row_attention_vectors.append(vec)
                continue

            spans = [s.strip() for s in annot_rationale.split(",") if s.strip()]
            if not spans:
                row_attention_vectors.append(vec)
                continue

            for span_text in spans:
                start = 0
                while True:
                    idx = text.find(span_text, start)
                    if idx < 0:
                        break
                    span_start, span_end = idx, idx + len(span_text)
                    for vec_idx, tok_idx in enumerate(real_token_indices):
                        tok_start, tok_end = offsets[tok_idx]
                        if tok_end > span_start and tok_start < span_end:
                            vec[vec_idx] = 1.0
                    start = idx + 1

            row_attention_vectors.append(vec)

        all_attention_vectors.append(row_attention_vectors)

    # Write attention vectors to new columns
    for i in range(max_annotators):
        col_name = f"embert_attention_{i+1}"
        col_vectors = []
        for row_vecs, num_tokens in zip(all_attention_vectors, token_lengths):
            if i < len(row_vecs):
                vec_str = " ".join(f"{int(v)}" for v in row_vecs[i])
            else:
                vec_str = " ".join(["0"] * num_tokens)
            col_vectors.append(vec_str)
        df[col_name] = col_vectors

    return df

# ==============================
# Explanation class for FERRET
# ==============================
class Explanation:
    def __init__(self, text, tokens, scores, explainer, target):
        self.text = text
        self.tokens = tokens
        self.scores = np.array(scores, dtype=np.float32)
        self.explainer = explainer
        self.target = target

    def __repr__(self):
        return f"Explanation(text={self.text!r}, tokens={self.tokens}, scores=array({self.scores}, dtype=float32), explainer={self.explainer!r}, target={self.target})"

# ==============================
# DEVICE SETUP
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ==============================
# FERRET PIPELINE
# ==============================
for hf_model_name in hf_model_names:
    print(f"\n==============================")
    print(f"[INFO] Starting pipeline for model: {hf_model_name}")
    print(f"==============================")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
        use_safetensors=True
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
        use_safetensors=True
    )

    bench = Benchmark(model, tokenizer)

    df = generate_attention_vectors_from_rationales(df, tokenizer)

    # List of explainers you want to use
    explainer_names = [
        "Partition SHAP", "LIME", "Gradient", "Gradient (x Input)",
        "Integrated Gradient", "Integrated Gradient (x Input)"
    ]

    ferret_filename = f"{hf_model_name.replace('/', '_')}_ferret_plausibility.csv"
    header_written = os.path.exists(ferret_filename)

    # To ensure no empty cells: collect all possible output columns
    all_fieldnames = set(["Index", "Text", "final_label", "final_label_numeric", "Annotations", "Rationale"])

    # --- MAIN LOOP ---
    for idx in tqdm(range(len(df)), desc="FERRET (Plausibility Only)"):
        row = df.iloc[idx]

        ferret_row = {
            "Index": idx,
            "Text": row["Text"],
            "final_label": row["final_label"],
            "final_label_numeric": int(row["final_label_numeric"]),
            "Annotations": row.get("Annotations", ""),
            "Rationale": row.get("Rationale", ""),
        }

        # Prepare explanations for all explainers
        row_explanations = {}
        for explainer_name in explainer_names:
            score_col = f"{explainer_name}_ImportanceScores"
            tokens_col = "Tokens"
            if pd.notna(row.get(score_col)) and pd.notna(row.get(tokens_col)):
                try:
                    scores = [float(score) for score in str(row[score_col]).split()]
                    tokens = str(row[tokens_col]).split()
                    target_label = int(row["final_label_numeric"])
                    row_explanations[explainer_name] = Explanation(
                        text=row["Text"], tokens=tokens, scores=scores,
                        explainer=explainer_name, target=target_label
                    )
                except Exception as e:
                    print(f"Could not create explanation for explainer {explainer_name} at index {idx}: {e}")
                    continue

        # Discover available metrics for plausibility
        available_metrics = set()
        if row_explanations:
            first_explainer = next(iter(row_explanations.keys()))
            first_explanation = row_explanations[first_explainer]
            for test_annot_idx in range(3):
                test_attn_col = f"embert_attention_{test_annot_idx+1}"
                test_human_rationale_str = str(row.get(test_attn_col, ""))
                test_human_rationale = [int(v) for v in test_human_rationale_str.split() if v.isdigit()]
                if any(test_human_rationale):
                    try:
                        test_plaus_eval = bench.evaluate_explanations(
                            [first_explanation],
                            human_rationale=test_human_rationale,
                            target=first_explanation.target,
                            skip_faithfulness=True
                        )
                        if test_plaus_eval and len(test_plaus_eval) > 0:
                            test_eval_obj = test_plaus_eval[0]
                            if hasattr(test_eval_obj, "evaluation_scores") and test_eval_obj.evaluation_scores:
                                for score in test_eval_obj.evaluation_scores:
                                    if score.name in ['auprc_plau', 'token_f1_plau', 'token_iou_plau']:
                                        available_metrics.add(score.name)
                                break
                    except Exception as e:
                        print(f"Error discovering metrics with {first_explainer} and annotator {test_annot_idx+1}: {e}")
                        continue

        print(f"Row {idx}: Using FERRET plausibility metrics: {list(available_metrics)}")

        # --- Evaluate plausibility for each explainer/annotator combination ---
        for explainer_name in explainer_names:
            if explainer_name not in row_explanations:
                for annot_idx in range(3):
                    for metric in available_metrics:
                        colname = f"{explainer_name}Annotator{annot_idx+1}{metric}"
                        ferret_row[colname] = "N/A"
                        all_fieldnames.add(colname)
                continue

            explanation = row_explanations[explainer_name]
            label = explanation.target

            for annot_idx in range(3):
                attn_col = f"embert_attention_{annot_idx+1}"
                human_rationale_str = str(row.get(attn_col, ""))
                human_rationale = [int(v) for v in human_rationale_str.split() if v.isdigit()]

                annot_labels_list = str(row["Annotations"]).split("|")
                if annot_idx < len(annot_labels_list):
                    annot_label_str = annot_labels_list[annot_idx].split("-")[0].strip()
                else:
                    annot_label_str = ""

                final_label_str = inv_label_map[label]
                for metric in available_metrics:
                    colname = f"{explainer_name}Annotator{annot_idx+1}{metric}"
                    all_fieldnames.add(colname)

                if annot_label_str != final_label_str:
                    for metric in available_metrics:
                        ferret_row[f"{explainer_name}Annotator{annot_idx+1}{metric}"] = "N/A"
                    continue

                if any(human_rationale):
                    try:
                        plaus_eval = bench.evaluate_explanations(
                            [explanation],
                            human_rationale=human_rationale,
                            target=label,
                            skip_faithfulness=True
                        )
                        if plaus_eval and len(plaus_eval) > 0:
                            eval_obj = plaus_eval[0]
                            if hasattr(eval_obj, "evaluation_scores") and eval_obj.evaluation_scores:
                                for score in eval_obj.evaluation_scores:
                                    if score.name in ['auprc_plau', 'token_f1_plau', 'token_iou_plau']:
                                        ferret_row[f"{explainer_name}Annotator{annot_idx+1}{score.name}"] = float(score.score)
                    except Exception as e:
                        print(f"Error evaluating {explainer_name} for annotator {annot_idx+1} at index {idx}: {e}")
                        for metric in available_metrics:
                            ferret_row[f"{explainer_name}Annotator{annot_idx+1}{metric}"] = "N/A"
                else:
                    for metric in available_metrics:
                        ferret_row[f"{explainer_name}Annotator{annot_idx+1}{metric}"] = "N/A"

        # --- Ensure no empty cells: fill missing columns with "N/A" ---
        for col in all_fieldnames:
            if col not in ferret_row:
                ferret_row[col] = "N/A"

        # === SAVE THIS ROW TO CSV IMMEDIATELY ===
        write_header = not header_written
        with open(ferret_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_fieldnames))
            if write_header:
                writer.writeheader()
                header_written = True
            writer.writerow(ferret_row)

    print(f"[INFO] FERRET plausibility results saved row-wise to {ferret_filename}.")

    # --- Memory cleanup ---
    print(f"[INFO] Cleaning up memory for model {hf_model_name}...")
    del bench, model, tokenizer, df
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ==============================
# End of pipeline
# ==============================
