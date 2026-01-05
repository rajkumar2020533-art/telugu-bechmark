import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIG ---
args_dict = {
    "batch_size": 64,
    "num_epochs": 4,
    "learning_rate": 2e-5,
    "max_length": 128,
    "model_name": "bert-base-multilingual-cased",#replace with your model_name
    "num_labels": 3,
    "save_dir": "./saved_model"
}
os.makedirs(args_dict["save_dir"], exist_ok=True)

# --- LABEL MAPPING ---
label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
label2name = {v: k for k, v in label_mapping.items()}
label_ids = list(label2name.keys())

# --- LOAD DATA ---
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")
emoji_df = pd.read_csv("emoji.csv")

# --- FILTER INVALID LABELS ---
train_df = train_df[train_df["final_label"].isin(label_mapping)]
val_df = val_df[val_df["final_label"].isin(label_mapping)]
test_df = test_df[test_df["final_label"].isin(label_mapping)]

# --- TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(args_dict["model_name"])
emoji_list = emoji_df["emoji"].dropna().astype(str).str.strip().tolist()
emoji_set = set(emoji_list) - set(tokenizer.vocab.keys())
if emoji_set:
    tokenizer.add_tokens(list(emoji_set))
    print(f"Added {len(emoji_set)} emojis to tokenizer.")

# --- MODEL ---
model = AutoModelForSequenceClassification.from_pretrained(
    args_dict["model_name"], num_labels=args_dict["num_labels"]
)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- DATASET ---
class SimpleTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["Content"]
        label = label_mapping[row["final_label"]]
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long),
            text
        )

# --- DATALOADERS ---
train_loader = DataLoader(SimpleTextDataset(train_df, tokenizer), batch_size=args_dict["batch_size"], shuffle=True)
val_loader = DataLoader(SimpleTextDataset(val_df, tokenizer), batch_size=args_dict["batch_size"])
test_loader = DataLoader(SimpleTextDataset(test_df, tokenizer), batch_size=args_dict["batch_size"])

# --- TRAINING ---
optimizer = Adam(model.parameters(), lr=args_dict["learning_rate"])
val_metrics_history = []

for epoch in range(1, args_dict["num_epochs"] + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attn_mask, labels, _ = [x.to(device) for x in batch[:3]]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # --- VALIDATION ---
    model.eval()
    val_preds, val_labels, val_loss = [], [], 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attn_mask, labels, _ = [x.to(device) for x in batch[:3]]
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            val_preds.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_loss += outputs.loss.item()
    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="weighted")
    try:
        val_auroc = roc_auc_score(
            pd.get_dummies(val_labels), pd.get_dummies(val_preds),
            average="weighted", multi_class="ovo"
        )
    except:
        val_auroc = float("nan")

    # --- Label-wise Metrics ---
    prec, rec, f1, supp = precision_recall_fscore_support(val_labels, val_preds, labels=[0,1,2])
    labelwise = {}
    for i in [0, 1, 2]:
        idx = np.array(val_labels) == i
        if idx.sum() > 0:
            acc = (np.array(val_preds)[idx] == i).sum() / idx.sum()
        else:
            acc = 0.0
        labelwise[label2name[i]] = {
            "val_acc": acc,
            "val_f1": f1[i],
            "val_precision": prec[i],
            "val_recall": rec[i],
            "val_support": supp[i]
        }

    val_metrics_history.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_auroc": val_auroc,
        **{f"{label}_{m}": labelwise[label][m]
           for label in labelwise for m in labelwise[label]}
    })

    print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | AUROC={val_auroc:.4f}")

model.save_pretrained(args_dict["save_dir"])
tokenizer.save_pretrained(args_dict["save_dir"])
print(f"Last model saved after epoch {args_dict['num_epochs']}")
# --- SAVE VAL METRICS ---
pd.DataFrame(val_metrics_history).to_csv("val_metrics_detailed.csv", index=False)

# --- LOAD BEST MODEL ---
model = AutoModelForSequenceClassification.from_pretrained(args_dict["save_dir"]).to(device)
tokenizer = AutoTokenizer.from_pretrained(args_dict["save_dir"])

# --- TEST EVAL ---
model.eval()
all_preds, all_labels, all_sentences, all_tokens = [], [], [], []
test_loss = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attn_mask, labels, sentences = batch
        input_ids, attn_mask, labels = input_ids.to(device), attn_mask.to(device), labels.to(device)
        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
        test_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_sentences.extend(sentences)
        all_tokens.extend(tokenizer.batch_decode(input_ids.cpu(), skip_special_tokens=True))

test_loss /= len(test_loader)
test_acc = accuracy_score(all_labels, all_preds)
test_f1 = f1_score(all_labels, all_preds, average="weighted")
try:
    test_auroc = roc_auc_score(pd.get_dummies(all_labels), pd.get_dummies(all_preds), average="weighted", multi_class="ovo")
except:
    test_auroc = float("nan")

# --- LABEL-WISE TEST METRICS ---
prec, rec, f1, supp = precision_recall_fscore_support(all_labels, all_preds, labels=[0,1,2])
label_metrics = {
    "Label": [], "Accuracy": [], "F1": [], "Precision": [], "Recall": [], "Support": []
}
for i in [0, 1, 2]:
    idx = np.array(all_labels) == i
    if idx.sum() > 0:
        acc = (np.array(all_preds)[idx] == i).sum() / idx.sum()
    else:
        acc = 0.0
    label_name = label2name[i]
    label_metrics["Label"].append(label_name)
    label_metrics["Accuracy"].append(acc)
    label_metrics["F1"].append(f1[i])
    label_metrics["Precision"].append(prec[i])
    label_metrics["Recall"].append(rec[i])
    label_metrics["Support"].append(supp[i])
pd.DataFrame(label_metrics).to_csv("labelwise_test_metrics.csv", index=False)

# --- OVERALL TEST METRICS CSV ---
pd.DataFrame([{
    "Test Loss": test_loss,
    "Test Accuracy": test_acc,
    "Test F1 Score": test_f1,
    "Test AUROC": test_auroc
}]).to_csv("overall_test_metrics.csv", index=False)

# --- TEST PREDICTIONS ---
pd.DataFrame({
    "Content": all_sentences,
    "Tokens": all_tokens,
    "final_label": [label2name[l] for l in all_labels],
    "predicted_label": [label2name[p] for p in all_preds]
}).to_csv("test_predictions.csv", index=False)

# --- CONFUSION MATRIX ---
conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
conf_matrix_df = pd.DataFrame(conf_matrix, index=[label2name[i] for i in [0,1,2]],
                              columns=[label2name[i] for i in [0,1,2]])
conf_matrix_df.to_csv("confusion_matrix.csv")

# --- CONFUSION MATRIX PLOT ---
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# --- DONE ---
print("\n=== FINAL TEST METRICS ===")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test F1       : {test_f1:.4f}")
print(f"Test AUROC    : {test_auroc:.4f}")
print("All test metrics, predictions, and confusion matrix saved.")
