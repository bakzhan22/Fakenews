import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import re as _re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm


DEFAULTS = dict(
    data_a         = "final_dataset_for_advisor.csv",
    data_b         = "phd_training_dataset_qwen.csv",
    model_name     = "intfloat/multilingual-e5-large",
    max_len        = 256,
    epochs         = 5,
    batch_size     = 8,
    lr             = 1e-5,
    weight_decay   = 0.01,
    num_workers    = 0,
    seed           = 42,
    merge_strategy = "best",
    save_path      = "advisor_model_best.pt",
    history_path   = "training_history.csv",
)

_VERDICT = _re.compile(r"Вердикт\s*:\s*(TRUE|FALSE|UNVERIFIABLE)\.?\s*", _re.IGNORECASE)
_TOKENS  = _re.compile(r"\b(TRUE|FALSE|UNVERIFIABLE)\b", _re.IGNORECASE)
_STARRED = _re.compile(r"\*\*(real|fake|реальн\w*|фейк\w*|TRUE|FALSE)\*\*", _re.IGNORECASE)

def _clean_evidence(t):
    if not isinstance(t, str): return ""
    t = _VERDICT.sub("", t)
    t = _TOKENS.sub("", t)
    t = _STARRED.sub("", t)
    return t.strip()


def prepare_dataframe(data_a: str, data_b: str, strategy: str) -> pd.DataFrame:
    df_a = pd.read_csv(data_a)
    df_b = pd.read_csv(data_b)

    if strategy == "best
    ":
        df = pd.DataFrame({
            "claim"                  : df_a["claim"],
            "evidence_explanation"   : df_b["evidence_explanation"].apply(_clean_evidence),
            "commonsense_explanation": df_a["commonsense_explanation"],
            "textual_explanation"    : df_a["textual_explanation"],
            "label"                  : df_a["label"],
        })

    df = df.dropna(subset=["claim", "evidence_explanation", "label"])
    df = df[df["evidence_explanation"].str.len() > 10]
    df["commonsense_explanation"] = df["commonsense_explanation"].fillna("").astype(str)
    df["textual_explanation"]     = df["textual_explanation"].fillna("").astype(str)
    df = df.reset_index(drop=True)
    print(df["label"].value_counts().to_string())
    return df


class AdvisorDataset(Dataset):
    COLS = [
        "claim",
        "evidence_explanation",
        "commonsense_explanation",
        "textual_explanation",
    ]
    E5_PREFIXES = {
        "claim"                  : "query: ",
        "evidence_explanation"   : "passage: ",
        "commonsense_explanation": "passage: ",
        "textual_explanation"    : "passage: ",
    }

    def __init__(self, df, tokenizer, max_len, use_e5_prefix=True):
        self.df           = df.reset_index(drop=True)
        self.tokenizer    = tokenizer
        self.max_len      = max_len
        self.use_e5_prefix = use_e5_prefix

    def _encode(self, text, col_name=""):
        if self.use_e5_prefix:
            prefix = self.E5_PREFIXES.get(col_name, "passage: ")
            text   = prefix + (str(text) if pd.notna(text) else "")
        else:
            text   = str(text) if pd.notna(text) else ""
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            [self._encode(row[c], c) for c in self.COLS],
            torch.tensor(int(row["label"]), dtype=torch.long),
        )


def collate_fn(batch):
    encodings_list, labels = zip(*batch)
    n = len(encodings_list[0])
    batched = [
        {k: torch.stack([encodings_list[b][i][k] for b in range(len(encodings_list))])
         for k in encodings_list[0][i]}
        for i in range(n)
    ]
    return batched, torch.stack(labels)

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(hidden_dim, num_heads,
                                             dropout=dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, query, context):
        attn_out, weights = self.attn(query, context, context)
        query = self.norm1(query + self.dropout(attn_out))
        query = self.norm2(query + self.dropout(self.ffn(query)))
        return query, weights


class AdvisorModel(nn.Module):
    
    def __init__(self, model_name, num_classes=2, num_heads=8,
                 dropout=0.1, freeze_layers=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size 

        if freeze_layers > 0:
            modules_to_freeze = [
                self.encoder.embeddings,
                *self.encoder.encoder.layer[:freeze_layers]
            ]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        self.cross_attn_c2e = CrossAttentionLayer(H, num_heads, dropout)
        self.cross_attn_e2c = CrossAttentionLayer(H, num_heads, dropout)

        self.proj_db = nn.Linear(H, H)
        self.proj_cs = nn.Linear(H, H)
        self.proj_tx = nn.Linear(H, H)

        self.classifier = nn.Sequential(
            nn.LayerNorm(H * 4),
            nn.Linear(H * 4, H * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(H * 2),
            nn.Linear(H * 2, H),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(H, num_classes),
        )

    def _cls(self, enc):
        out = self.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        return out.last_hidden_state[:, 0, :]

    def forward(self, encodings):
        claim_emb = self._cls(encodings[0])             

        e_db = torch.tanh(self.proj_db(self._cls(encodings[1])))  
        e_cs = torch.tanh(self.proj_cs(self._cls(encodings[2])))  
        e_tx = torch.tanh(self.proj_tx(self._cls(encodings[3])))  

        context = torch.stack([e_db, e_cs, e_tx], dim=1)   

        query = claim_emb.unsqueeze(1)                      
        attended_c, _ = self.cross_attn_c2e(query, context)
        attended_c = attended_c.squeeze(1)                   

        attended_e, _ = self.cross_attn_e2c(context, query.expand(-1, 3, -1))
        attended_e = attended_e.mean(dim=1)             

        exp_max, _ = context.max(dim=1)                    

        combined = torch.cat([claim_emb, attended_c, attended_e, exp_max], dim=-1)
        return self.classifier(combined)


def run_train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    pbar = tqdm(loader, desc="  train", leave=False, ncols=80)
    for encodings, labels in pbar:
        encodings = [{k: v.to(device) for k, v in e.items()} for e in encodings]
        labels    = labels.to(device)
        optimizer.zero_grad()
        logits = model(encodings)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for encodings, labels in loader:
        encodings = [{k: v.to(device) for k, v in e.items()} for e in encodings]
        labels    = labels.to(device)
        logits    = model(encodings)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc             = accuracy_score(all_labels, all_preds)
    prec, rec, f1,_ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    report = classification_report(
        all_labels, all_preds,
        target_names=["disagree (0)", "agree (1)"], zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)
    return total_loss / len(loader), acc, prec, rec, f1, report, cm


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  Device : {device}  |  Backbone: {args.model_name}")
    print(f"  Epochs : {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}")
    print(f"{'='*65}")

    df = prepare_dataframe(args.data_a, args.data_b, args.merge_strategy)

    train_df, tmp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=args.seed
    )
    val_df, test_df = train_test_split(
        tmp, test_size=0.50, stratify=tmp["label"], random_state=args.seed
    )
    print(f"\n  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}\n")

    print("  Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    use_e5 = "e5" in args.model_name.lower()
    if use_e5:
        print(" E5")

    def make_loader(d, shuffle):
        return DataLoader(
            AdvisorDataset(d, tokenizer, args.max_len, use_e5_prefix=use_e5),
            batch_size=args.batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=args.num_workers,
        )

    train_loader = make_loader(train_df, True)
    val_loader   = make_loader(val_df,   False)
    test_loader  = make_loader(test_df,  False)

    print("load. AdvisorModel")
    model = AdvisorModel(
        args.model_name,
        freeze_layers=args.freeze_layers
    ).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.1, anneal_strategy="cos",
    )
    if args.class_weight > 1.0:
        weights = torch.tensor([args.class_weight, 1.0]).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=args.label_smoothing
        )
        print(f"  Class weights: disagree={args.class_weight} agree=1.0")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f" param: {trainable:,} обучаемых / {total:,} всего")
    print(f"  Freeze layers: {args.freeze_layers}  |  Label smoothing: {args.label_smoothing}\n")

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  param: {n:,}\n")

    best_f1 = 0.0
    history = []
    print(f"{'─'*65}")
    print(f"{'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  |  "
          f"{'VlLoss':>8}  {'VlAcc':>7}  {'F1':>7}  {'P':>7}  {'R':>7}")
    print(f"{'─'*65}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_train(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_p, vl_r, vl_f1, _, _ = run_eval(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history.append(dict(
            epoch=epoch,
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=vl_loss,   val_acc=vl_acc,
            val_f1=vl_f1,       val_prec=vl_p, val_rec=vl_r,
        ))

        star = ""
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            torch.save(model.state_dict(), args.save_path)
            star = "  ★"

        print(f"{epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  |  "
              f"{vl_loss:>8.4f}  {vl_acc:>7.4f}  {vl_f1:>7.4f}  "
              f"{vl_p:>7.4f}  {vl_r:>7.4f}{star}")

    print(f"\n{'='*65}")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    _, ts_acc, ts_p, ts_r, ts_f1, ts_report, ts_cm = run_eval(
        model, test_loader, criterion, device
    )
    print(f"\n Acc={ts_acc:.4f}  F1={ts_f1:.4f}  "
          f"Prec={ts_p:.4f}  Rec={ts_r:.4f}")
    print(f"\n{ts_report}")
    print(f"  Confusion matrix:\n{ts_cm}")
    print(f"{'='*65}\n")
    return model, tokenizer

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Advisor Model — COLING 2025",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_a",         default=DEFAULTS["data_a"])
    p.add_argument("--data_b",         default=DEFAULTS["data_b"])
    p.add_argument("--merge_strategy", default=DEFAULTS["merge_strategy"],
                   choices=["best", "a_only", "b_only", "augment"])
    p.add_argument("--model_name",     default=DEFAULTS["model_name"])
    p.add_argument("--max_len",        type=int,   default=DEFAULTS["max_len"])
    p.add_argument("--batch_size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight_decay",   type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--num_workers",    type=int,   default=DEFAULTS["num_workers"])
    p.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])
    p.add_argument("--save_path",      default=DEFAULTS["save_path"])
    p.add_argument("--history_path",   default=DEFAULTS["history_path"])
    p.add_argument("--freeze_layers",  type=int,   default=6,
                   help="Сколько нижних слоёв RoBERTa заморозить (0=нет)")
    p.add_argument("--class_weight",   type=float, default=1.5,
                   help="Вес класса disagree (>1 = больше внимания фейкам)")
    p.add_argument("--label_smoothing",type=float, default=0.1,
                   help="Label smoothing для CrossEntropyLoss")
    main(p.parse_args())