import os
import argparse
import requests
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

load_dotenv()

OLLAMA_URL  = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
MODEL_NAME  = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
CHROMA_DIR  = os.getenv("CHROMA_DIR",  "chromadb_gemini")
COLLECTION  = os.getenv("CHROMA_COLLECTION", "gov_news_gemini")

# ── Промпты ──────────────────────────────────────────────────────────

CS_PROMPT = """Ты — система анализа новостей. Оцени логику и правдоподобность заголовка, используя только здравый смысл.
Не используй внешние факты. Не выноси вердикт — только анализируй внутреннюю логику и правдоподобность.
Заголовок: {claim}
Вывод (2-3 предложения, русский язык):"""

TX_PROMPT = """Ты — система анализа текста. Проанализируй текстовые и стилистические характеристики заголовка.
Не оценивай правдивость. Только описывай стиль, наличие кликбейта, эмоций, манипуляций.
Заголовок: {claim}
Текстовое описание (2-3 предложения, русский язык):"""

DB_PROMPT = """Ты — система проверки фактов. Проверь заголовок на основе текста статьи.
Не выноси вердикт — только описывай что говорится в статье по теме заголовка.
Заголовок: {claim}
Текст статьи: {body}
Фактическое обоснование (2-3 предложения, русский язык):"""


def call_ollama(prompt: str) -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME, "prompt": prompt, "stream": False
        }, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"Ошибка: {e}"


def init_rag():
    try:
        from bmwGpt2 import ChromaDBManager, TerminalRAG
        db    = ChromaDBManager(CHROMA_DIR)
        store = db.create_or_load_db(collection_name=COLLECTION)
        rag   = TerminalRAG(store)
        rag.invalidate_cache()
        size  = len(store.get().get("ids", []))
        print(f"ChromaDB: {size} чанков в '{COLLECTION}'")
        return rag
    except Exception as e:
        print(f"ChromaDB недоступна: {e}")
        return None


def get_evidence_from_rag(rag, query: str, fallback: str) -> str:
    """RAG поиск. Если не найдено — fallback на articleBody."""
    if rag is not None:
        try:
            docs, _, _ = rag.retrieve_hybrid(query)
            if docs:
                return "\n\n".join(
                    f"[{i+1}] {doc.page_content[:300].strip()}"
                    for i, doc in enumerate(docs[:3])
                )
        except Exception:
            pass
    # fallback: используем сам articleBody как доказательство
    return fallback[:600]


# ── Архитектура (точная копия из advisor_train.py) ───────────────────

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(hidden_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )
    def forward(self, q, c):
        o, w = self.attn(q, c, c)
        q = self.norm1(q + self.drop(o))
        q = self.norm2(q + self.drop(self.ffn(q)))
        return q, w


class AdvisorModel(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        H = self.encoder.config.hidden_size
        self.cross_attn_c2e = CrossAttentionLayer(H, dropout=dropout)
        self.cross_attn_e2c = CrossAttentionLayer(H, dropout=dropout)
        self.proj_db = nn.Linear(H, H)
        self.proj_cs = nn.Linear(H, H)
        self.proj_tx = nn.Linear(H, H)
        self.classifier = nn.Sequential(
            nn.LayerNorm(H*4), nn.Linear(H*4, H*2), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(H*2), nn.Linear(H*2, H),   nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(H, num_classes),
        )

    def _cls(self, enc):
        return self.encoder(**enc).last_hidden_state[:, 0, :]

    def forward(self, encs):
        c  = self._cls(encs[0])
        db = torch.tanh(self.proj_db(self._cls(encs[1])))
        cs = torch.tanh(self.proj_cs(self._cls(encs[2])))
        tx = torch.tanh(self.proj_tx(self._cls(encs[3])))
        ctx = torch.stack([db, cs, tx], dim=1)
        q = c.unsqueeze(1)
        ac, _ = self.cross_attn_c2e(q, ctx)
        ac = ac.squeeze(1)
        ae, _ = self.cross_attn_e2c(ctx, q.expand(-1, 3, -1))
        ae = ae.mean(1)
        xm, _ = ctx.max(1)
        return self.classifier(torch.cat([c, ac, ae, xm], -1))


# ── Predictor ────────────────────────────────────────────────────────

class AdvisorPredictor:
    E5_PREFIXES = {
        "claim"                  : "query: ",
        "evidence_explanation"   : "passage: ",
        "commonsense_explanation": "passage: ",
        "textual_explanation"    : "passage: ",
    }

    def __init__(self, checkpoint: str,
                 model_name: str = "intfloat/multilingual-e5-large",
                 max_len: int = 256):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = AdvisorModel(model_name).to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device)
        )
        self.model.eval()
        self.max_len    = max_len
        self.model_name = model_name
        print(f"Модель загружена: {checkpoint} → {self.device}")

    def _enc(self, text: str, col: str = "") -> dict:
        if "e5" in self.model_name.lower():
            text = self.E5_PREFIXES.get(col, "passage: ") + str(text)
        e = self.tokenizer(str(text), max_length=self.max_len,
                           padding="max_length", truncation=True,
                           return_tensors="pt")
        return {k: v.to(self.device) for k, v in e.items()}

    @torch.no_grad()
    def predict(self, claim, e_db, e_cs, e_tx) -> dict:
        encs  = [
            self._enc(claim, "claim"),
            self._enc(e_db,  "evidence_explanation"),
            self._enc(e_cs,  "commonsense_explanation"),
            self._enc(e_tx,  "textual_explanation"),
        ]
        probs = torch.softmax(self.model(encs), dim=-1)[0]
        label = int(probs.argmax().item())
        return {
            "pred_label"   : label,
            "pred_veracity": "agree" if label == 1 else "disagree",
            "prob_disagree": round(probs[0].item(), 4),
            "prob_agree"   : round(probs[1].item(), 4),
        }


# ── Main ─────────────────────────────────────────────────────────────

def main(args):
    # Загрузка и merge
    stances = pd.read_csv(args.stances)
    bodies  = pd.read_csv(args.bodies)

    df = stances.merge(
        bodies[["Body ID", "articleBody"]],
        on="Body ID", how="left"
    )
    df = df.dropna(subset=["Headline1", "articleBody"]).reset_index(drop=True)

    # Срез для параллельного запуска
    if args.start is not None or args.end is not None:
        start = args.start or 0
        end   = args.end   or len(df)
        df    = df.iloc[start:end].reset_index(drop=True)
        print(f"Срез: [{start}:{end}] — {len(df)} записей")

    if args.limit:
        df = df.head(args.limit)

    print(f"Записей     : {len(df)}")
    print(f"label=1     : {df['label'].sum()}")
    print(f"label=0     : {(df['label']==0).sum()}")

    # RAG
    rag = init_rag()

    # Генерация объяснений
    print(f"\nГенерация объяснений (ê_cs + ê_tx + ê_db)...")
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Генерация"):
        claim = str(row["Headline1"]).strip()
        body  = str(row["articleBody"]).strip()[:800]
        label = int(row["label"])

        e_cs = call_ollama(CS_PROMPT.format(claim=claim))
        e_tx = call_ollama(TX_PROMPT.format(claim=claim))

        # ê_db: RAG если доступен, иначе articleBody
        evidence = get_evidence_from_rag(rag, claim, body)
        e_db = call_ollama(DB_PROMPT.format(claim=claim, body=evidence))

        rows.append({
            "Body ID"                : row["Body ID"],
            "claim"                  : claim,
            "articleBody"            : body,
            "evidence_explanation"   : e_db,
            "commonsense_explanation": e_cs,
            "textual_explanation"    : e_tx,
            "label"                  : label,
        })

    exp_df = pd.DataFrame(rows)

    # Предсказание
    print(f"\nПредсказание Advisor Model...")
    predictor = AdvisorPredictor(
        checkpoint  = args.checkpoint,
        model_name  = args.model_name,
        max_len     = args.max_len,
    )

    preds = []
    for _, row in tqdm(exp_df.iterrows(), total=len(exp_df), desc="Предсказание"):
        result = predictor.predict(
            claim = row["claim"],
            e_db  = row["evidence_explanation"],
            e_cs  = row["commonsense_explanation"],
            e_tx  = row["textual_explanation"],
        )
        preds.append(result)

    result_df = exp_df.copy()
    result_df["pred_label"]    = [p["pred_label"]     for p in preds]
    result_df["pred_veracity"] = [p["pred_veracity"]  for p in preds]
    result_df["prob_disagree"] = [p["prob_disagree"]  for p in preds]
    result_df["prob_agree"]    = [p["prob_agree"]     for p in preds]

    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    # ── Метрики ───────────────────────────────────────────────────────
    y_true = result_df["label"].tolist()
    y_pred = result_df["pred_label"].tolist()

    acc          = accuracy_score(y_true, y_pred)
    p, r, f1, _  = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred,
        target_names=["disagree (0)", "agree (1)"], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f"  ФИНАЛЬНЫЕ МЕТРИКИ — TEST SET ({len(result_df)} записей)")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {p:.4f}  (macro)")
    print(f"  Recall    : {r:.4f}  (macro)")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"\n{report}")
    print(f"  Confusion matrix:\n{cm}")
    print(f"{'='*55}")
    print(f"\nСохранено: {args.output}")
    print(f"\nРаспределение предсказаний:")
    print(result_df["pred_veracity"].value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stances",    default="fakenews/test_stances_unlebeledb.csv")
    p.add_argument("--bodies",     default="fakenews/test_bodies.csv")
    p.add_argument("--checkpoint", default="advisor_model_best.pt")
    p.add_argument("--model_name", default="intfloat/multilingual-e5-large")
    p.add_argument("--max_len",    type=int, default=256)
    p.add_argument("--output",     default="test_predictions.csv")
    p.add_argument("--limit",      type=int, default=None)
    p.add_argument("--start",      type=int, default=None, help="Начальный индекс")
    p.add_argument("--end",        type=int, default=None, help="Конечный индекс")
    main(p.parse_args())