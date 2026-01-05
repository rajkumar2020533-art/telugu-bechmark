import os
import torch
import random
import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import Adam
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support, accuracy_score

# Reproducibility
def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(13)

# Configurations
param_grid = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
    "batch_size": [16, 32, 64]
}
num_epochs = 10
max_length = 128
model_name = "bert-base-multilingual-cased"
num_labels = 3

# Tokenizer + Emoji Extension
emoji_df = pd.read_csv("emoji.csv")
emoji_list = emoji_df.iloc[:, 0].dropna().astype(str).unique().tolist()

tokenizer = BertTokenizer.from_pretrained(model_name)
new_tokens = list(set(emoji_list) - set(tokenizer.vocab.keys()))
if new_tokens:
    tokenizer.add_tokens(new_tokens)
    print(f"Added {len(new_tokens)} emojis to tokenizer.")

# Data loading
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

valid_labels = {"Negative": 0, "Neutral": 1, "Positive": 2}
train_df = train_df[train_df["final_label"].isin(valid_labels)]
val_df = val_df[val_df["final_label"].isin(valid_labels)]

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["Content"]
        label = valid_labels[row["final_label"]]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long)
        )

train_dataset = CustomDataset(train_df, tokenizer, max_length)
val_dataset = CustomDataset(val_df, tokenizer, max_length)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Results directory and file
os.makedirs("results", exist_ok=True)
results_path = "results/grid_search_metrics.csv"

if not os.path.exists(results_path):
    with open(results_path, "w") as f:
        f.write("timestamp,learning_rate,batch_size,epoch,val_macro_f1,val_auroc,"
                "acc_negative,prec_negative,rec_negative,f1_negative,"
                "acc_neutral,prec_neutral,rec_neutral,f1_neutral,"
                "acc_positive,prec_positive,rec_positive,f1_positive\n")

# Hyperparameter grid search
for lr, bs in itertools.product(param_grid["learning_rate"], param_grid["batch_size"]):
    print(f"\nStarting config: LR={lr}, Batch Size={bs}")
    set_seed(13)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    if new_tokens:
        model.resize_token_embeddings(len(tokenizer))

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        val_preds, val_probs, val_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                logits = model(input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = torch.argmax(logits, axis=1).cpu().tolist()

                val_probs.extend(probs)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().tolist())

        val_macro_f1 = f1_score(val_labels, val_preds, average="macro")
        val_auroc = roc_auc_score(
            np.eye(num_labels)[val_labels],
            np.array(val_probs),
            average="macro",
            multi_class="ovr"
        )

        # Label-wise metrics
        report = precision_recall_fscore_support(val_labels, val_preds, labels=[0, 1, 2], zero_division=0)
        acc_per_label = []
        for i in range(num_labels):
            idx = np.array(val_labels) == i
            correct = (np.array(val_preds)[idx] == i).sum()
            total = idx.sum()
            acc = correct / total if total > 0 else 0
            acc_per_label.append(acc)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            timestamp, lr, bs, epoch, f"{val_macro_f1:.4f}", f"{val_auroc:.4f}"
        ]
        for i in range(num_labels):
            row.extend([
                f"{acc_per_label[i]:.4f}",
                f"{report[0][i]:.4f}",  # precision
                f"{report[1][i]:.4f}",  # recall
                f"{report[2][i]:.4f}"   # f1
            ])

        with open(results_path, "a") as f:
            f.write(",".join(map(str, row)) + "\n")

        print(f"[Epoch {epoch}] LR={lr}, BS={bs} | F1={val_macro_f1:.4f} | AUROC={val_auroc:.4f}")

print(f"\nGrid Search Complete. Results saved to: {results_path}")
