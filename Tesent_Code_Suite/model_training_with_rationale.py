import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_fscore_support
import warnings
import random

def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

set_seed(13)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIG ---
model_name = "bert-base-multilingual-cased"   # Set your model name here
num_epochs = 4
max_length = 128
num_labels = 3
learning_rate = 2e-5
batch_size = 64
optimizer_type = "Adam"
lambda_attn = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD DATA ---
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")
valid_labels = {"Negative": 0, "Neutral": 1, "Positive": 2}
train_df = train_df[train_df["final_label"].isin(valid_labels.keys())]
val_df = val_df[val_df["final_label"].isin(valid_labels.keys())]
test_df = test_df[test_df["final_label"].isin(valid_labels.keys())]
if train_df.empty:
    raise ValueError("Train dataset empty after filtering.")
if val_df.empty:
    raise ValueError("Validation dataset empty after filtering.")

# --- FUNCTIONS ---
def generate_attention_vectors_from_rationales(df, tokenizer, epsilon=1e-8):
    attention_vectors = []
    for _, row in df.iterrows():
        text = str(row["Content"])
        final_label = str(row["final_label"]).strip()
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]
        num_tokens = len(offsets)
        avg_vector = np.zeros(num_tokens, dtype=np.float32)
        annotations = str(row.get("Annotations", "")).split("|")
        rationales = str(row.get("Rationale", "")).split("|")
        annot_vectors = []
        for annot_label, annot_rationale in zip(annotations, rationales):
            if not annot_label:
                continue
            if annot_label.split("-")[0].strip() != final_label:
                continue
            spans = [s.strip() for s in annot_rationale.split(",") if s.strip()]
            if not spans:
                continue
            vec = np.zeros(num_tokens, dtype=np.float32)
            for span_text in spans:
                start = 0
                while True:
                    idx = text.find(span_text, start)
                    if idx < 0:
                        break
                    span_start, span_end = idx, idx + len(span_text)
                    for i, (tok_start, tok_end) in enumerate(offsets):
                        if tok_end > span_start and tok_start < span_end:
                            vec[i] = 1.0
                    start = idx + 1
            if vec.sum() > 0:
                annot_vectors.append(vec)
        if annot_vectors:
            avg_vector = np.mean(annot_vectors, axis=0)
            avg_vector = np.where(avg_vector == 0, epsilon, avg_vector)
        attn_str = " ".join(f"{v:.8f}" for v in avg_vector)
        attention_vectors.append(attn_str)
    df["embert_attention"] = attention_vectors
    return df

class RationaleDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128, label_mapping=None):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["Content"]
        label = self.label_mapping[row["final_label"]]
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        rationale_raw = [float(x) for x in row["embert_attention"].split()] \
            if pd.notna(row["embert_attention"]) and row["embert_attention"].strip() else []
        rationale_vector = np.concatenate([
            np.array([0.0], dtype=np.float32),
            np.array(rationale_raw, dtype=np.float32),
            np.array([0.0], dtype=np.float32)
        ])
        rationale_vector = rationale_vector[:self.max_length]
        if len(rationale_vector) < self.max_length:
            rationale_vector = np.pad(rationale_vector, (0, self.max_length - len(rationale_vector)), constant_values=0.0)
        rationale_tensor = torch.tensor(rationale_vector, dtype=torch.float32)
        if torch.sum(rationale_tensor) == 0.0:
            has_rationale = False
            rationale_probs = torch.ones(self.max_length, dtype=torch.float32) / self.max_length
        else:
            has_rationale = True
            rationale_probs = torch.softmax(rationale_tensor, dim=0)
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            rationale_probs,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(has_rationale, dtype=torch.bool)
        )

class RationaleModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, output_attentions=True)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        last_layer_attn = outputs.attentions[-1]  # (batch, heads, seq, seq)
        cls_attn = last_layer_attn[:, :, 0, :]    # (batch, heads, seq)
        cls_attn_avg = cls_attn.mean(dim=1)       # (batch, seq)
        return logits, cls_attn_avg

def evaluate_model(model, val_loader, criterion_cls, device, valid_labels, num_labels):
    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, _, labels, _ = [b.to(device) for b in batch]
            logits, _ = model(input_ids, attention_mask)
            loss = criterion_cls(logits, labels)
            total_val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    avg_val_loss = total_val_loss / len(val_loader)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)
    accuracy = accuracy_score(all_labels_np, all_preds_np)
    f1_macro = f1_score(all_labels_np, all_preds_np, average="macro")
    try:
        y_true_oh = np.eye(num_labels)[all_labels_np]
        auroc_ovr = roc_auc_score(y_true_oh, all_probs_np, multi_class="ovr")
    except:
        auroc_ovr = -1.0
    class_wise_metrics = {}
    target_names = sorted(valid_labels, key=valid_labels.get)
    label_indices = [valid_labels[label_name] for label_name in target_names]
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        all_labels_np, all_preds_np, labels=label_indices, average=None)
    for i, label_name in enumerate(target_names):
        label_id = valid_labels[label_name]
        class_wise_metrics[f"{label_name}_precision"] = precision[i]
        class_wise_metrics[f"{label_name}_recall"] = recall[i]
        class_wise_metrics[f"{label_name}_f1"] = f1_per_class[i]
        label_mask = all_labels_np == label_id
        correct_preds = np.sum((all_preds_np == label_id) & label_mask)
        total_label = np.sum(label_mask)
        if total_label > 0:
            class_wise_metrics[f"{label_name}_accuracy"] = correct_preds / total_label
        else:
            class_wise_metrics[f"{label_name}_accuracy"] = -1.0
        try:
            binary_labels = (all_labels_np == label_id).astype(int)
            class_probs = all_probs_np[:, label_id]
            if len(np.unique(binary_labels)) > 1:
                class_wise_metrics[f"{label_name}_auroc"] = roc_auc_score(binary_labels, class_probs)
            else:
                class_wise_metrics[f"{label_name}_auroc"] = -1.0
        except:
            class_wise_metrics[f"{label_name}_auroc"] = -1.0
    return avg_val_loss, accuracy, f1_macro, auroc_ovr, class_wise_metrics

def train_model(model, train_loader, val_loader, num_epochs, device, lambda_attn=1.0, optimizer=None, learning_rate=2e-5, results_writer=None, results_file_handle=None):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            input_ids, attention_mask, rationale_probs, labels, has_rationale = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits, model_attention = model(input_ids, attention_mask)
            loss_cls = criterion_cls(logits, labels)
            loss = loss_cls
            if has_rationale.any():
                model_attn_batch = model_attention[has_rationale]
                rationale_batch = rationale_probs[has_rationale]
                log_model_attn = torch.log(model_attn_batch + 1e-8)
                loss_kl = criterion_kl(log_model_attn, rationale_batch)
                loss += lambda_attn * loss_kl
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1_macro, val_auroc_ovr, class_wise_metrics = evaluate_model(model, val_loader, criterion_cls, device, valid_labels, num_labels)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 (Macro): {val_f1_macro:.4f} | Val AUROC (OvR): {val_auroc_ovr:.4f}")
        sorted_labels = sorted(valid_labels, key=valid_labels.get)
        for label_name in sorted_labels:
            print(f"  {label_name}: P={class_wise_metrics[f'{label_name}_precision']:.4f}, R={class_wise_metrics[f'{label_name}_recall']:.4f}, F1={class_wise_metrics[f'{label_name}_f1']:.4f}, Acc={class_wise_metrics[f'{label_name}_accuracy']:.4f}, AUROC={class_wise_metrics[f'{label_name}_auroc']:.4f}")

        if results_writer and results_file_handle:
            row_data = [
                learning_rate, 
                batch_size, 
                optimizer_type, 
                lambda_attn, 
                epoch + 1,
                avg_train_loss, 
                val_loss, 
                val_acc, 
                val_f1_macro, 
                val_auroc_ovr
            ]
            for label_name in sorted_labels:
                row_data.extend([
                    class_wise_metrics[f"{label_name}_precision"],
                    class_wise_metrics[f"{label_name}_recall"],
                    class_wise_metrics[f"{label_name}_f1"],
                    class_wise_metrics[f"{label_name}_accuracy"],
                    class_wise_metrics[f"{label_name}_auroc"]
                ])
            results_writer.writerow(row_data)
            results_file_handle.flush()
            os.fsync(results_file_handle.fileno())

# --- OUTPUT FOLDERS ---
csv_output_dir = "csv_outputs"
os.makedirs(csv_output_dir, exist_ok=True)
results_file = os.path.join(csv_output_dir, "results_detailed.csv")
headers = ["learning_rate", "batch_size", "optimizer", "lambda", "epoch", "train_loss", "val_loss", "val_accuracy", "val_f1_macro", "val_auroc_ovr"]
sorted_labels = sorted(valid_labels, key=valid_labels.get)
for label in sorted_labels:
    headers.extend([f"{label}_precision", f"{label}_recall", f"{label}_f1", f"{label}_accuracy", f"{label}_auroc"])

# --- INITIALIZE TOKENIZER & ADD EMOJIS ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
emoji_path = "emoji.csv"
if os.path.exists(emoji_path):
    emoji_df = pd.read_csv(emoji_path)
    emoji_list = emoji_df["emoji"].dropna().astype(str).str.strip().tolist()
    existing_vocab = set(tokenizer.get_vocab().keys())
    emoji_set = set(emoji_list) - existing_vocab
    if emoji_set:
        tokenizer.add_tokens(list(emoji_set))
        print(f"Added {len(emoji_set)} new emoji tokens to the tokenizer.")
    else:
        print("No new emojis to add.")
else:
    print(f"Emoji file not found at: {emoji_path}")

# --- PREPARE DATASETS ---
print("Generating attention vectors for training data...")
train_df_model = generate_attention_vectors_from_rationales(train_df.copy(), tokenizer)
print("Generating attention vectors for validation data...")
val_df_model = generate_attention_vectors_from_rationales(val_df.copy(), tokenizer)

train_dataset = RationaleDataset(train_df_model, tokenizer, max_length, label_mapping=valid_labels)
val_dataset = RationaleDataset(val_df_model, tokenizer, max_length, label_mapping=valid_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(13))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- CSV Setup ---
with open(results_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    model = RationaleModel(model_name=model_name, num_labels=num_labels).to(device)
    if 'emoji_set' in locals() and len(emoji_set) > 0:
        model.bert.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        lambda_attn=lambda_attn,
        optimizer=optimizer,
        learning_rate=learning_rate,
        results_writer=writer,
        results_file_handle=f
    )
    # Save final model and tokenizer
    model.bert.save_pretrained("model_outputs")
    tokenizer.save_pretrained("model_outputs")
    print(f"Final model and tokenizer saved to model_outputs")

