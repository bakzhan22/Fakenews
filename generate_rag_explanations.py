import re
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_URL   = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
COMMON_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
CHROMA_DIR   = os.getenv("CHROMA_DIR", "chromadb_gemini")
COLLECTION   = os.getenv("CHROMA_COLLECTION", "gov_news_gemini")


QUERY_PROMPT = """Ты — система проверки фактов. На основе заголовка новости сформулируй 3 поисковых запроса для поиска доказательств.

Заголовок: {claim}

Правила:
- Запросы должны быть краткими (5-10 слов)
- Убери имена собственные, оставь суть утверждения
- Сформулируй как вопросы для поиска в базе знаний

Ответь только списком запросов:
1. [запрос]
2. [запрос]
3. [запрос]"""

EVIDENCE_PROMPT = """Ты — система проверки фактов. Проверь заголовок новости на основе найденных доказательств.

Заголовок: {claim}

Поисковый запрос: {query}

Найденные доказательства:
{evidence}

На основе доказательств напиши краткое обоснование (2-3 предложения) на русском языке.
Только описывай факты из доказательств — не выноси вердикт."""


def call_ollama(prompt: str) -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": COMMON_MODEL, "prompt": prompt, "stream": False
        }, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"Ошибка: {e}"


def extract_first_query(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        clean = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if len(clean) > 5:
            return clean
    return text[:100]


def main(args):
    try:
        from bmwGpt2 import TerminalRAG, ChromaDBManager
        db_manager   = ChromaDBManager(CHROMA_DIR)
        vector_store = db_manager.create_or_load_db(collection_name=COLLECTION)
        rag          = TerminalRAG(vector_store)
        rag.invalidate_cache()
        size = len(vector_store.get().get("ids", []))
        print(f"ChromaDB: {size} чанков в '{COLLECTION}'")
        use_rag = True
    except Exception as e:
        print(f"ChromaDB недоступна ({e}), fallback на articleBody.")
        use_rag = False

    stances = pd.read_csv(args.stances)
    bodies  = pd.read_csv(args.bodies)

    df = stances.merge(
        bodies[["Body ID", "articleBody"]],
        on="Body ID", how="left"
    )
    df["label"] = (df["Stance"] == "agree").astype(int)
    df = df.dropna(subset=["Headline1", "articleBody"]).reset_index(drop=True)

    if args.limit:
        df = df.head(args.limit)

    print(f"Записей    : {len(df)}")
    print(f"agree  (1) : {df['label'].sum()}")
    print(f"disagree(0): {(df['label']==0).sum()}")
    print(f"Модель     : {COMMON_MODEL}\n")

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="RAG generation"):
        claim = str(row["Headline1"]).strip()
        body  = str(row["articleBody"]).strip()[:800]
        label = int(row["label"])

        try:
            query_text = call_ollama(QUERY_PROMPT.format(claim=claim))
            query      = extract_first_query(query_text)

            if use_rag:
                docs, _, _ = rag.retrieve_hybrid(query)
                parts = []
                for i, doc in enumerate(docs[:3]):
                    snippet = doc.page_content[:300].strip()
                    src     = doc.metadata.get("source_file", "unknown")
                    parts.append(f"[{i+1}] (источник: {src})\n{snippet}")
                evidence_text = "\n\n".join(parts) if parts else f"[1] {body}"
            else:
                evidence_text = f"[1] {body}"

            e_db = call_ollama(EVIDENCE_PROMPT.format(
                claim=claim, query=query, evidence=evidence_text
            ))

        except Exception as ex:
            e_db = f"Ошибка RAG: {ex}"

        rows.append({
            "claim"               : claim,
            "evidence_explanation": e_db,
            "label"               : label,
        })

    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"agree  (1) : {result['label'].sum()}")
    print(f"disagree(0): {(result['label']==0).sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stances", default="fakenews/train_stances.csv")
    p.add_argument("--bodies",  default="fakenews/train_bodies.csv")
    p.add_argument("--output",  default="phd_training_dataset_qwen.csv")
    main(p.parse_args())