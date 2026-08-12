import argparse
import requests
import pandas as pd
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:120b"

CS_PROMPT = """Ты — система анализа новостей. Оцени логику и правдоподобность заголовка, используя только здравый смысл.
Не используй внешние факты. Не выноси вердикт — только анализируй внутреннюю логику и правдоподобность.

Заголовок: {claim}

Вывод (2-3 предложения, русский язык):"""

TX_PROMPT = """Ты — система анализа текста. Проанализируй текстовые и стилистические характеристики заголовка.
Не оценивай правдивость и не используй внешние факты.
Только описывай: стиль (нейтральный/сенсационный), наличие кликбейта, эмоционального давления, манипуляций, КАПС, восклицаний.

Заголовок: {claim}

Текстовое описание (2-3 предложения, русский язык):"""


def call_ollama(prompt: str) -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME, "prompt": prompt, "stream": False
        }, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"Ошибка: {e}"


def main(args):
    stances = pd.read_csv(args.stances)
    bodies  = pd.read_csv(args.bodies)

    df = stances.merge(
        bodies[["Body ID", "articleBody"]],
        on="Body ID", how="left"
    )
    df["label"] = (df["Stance"] == "agree").astype(int)
    df = df.dropna(subset=["Headline1"]).reset_index(drop=True)

    if args.limit:
        df = df.head(args.limit)

    if args.start is not None or args.end is not None:
        start = args.start or 0
        end   = args.end   or len(df)
        df    = df.iloc[start:end].reset_index(drop=True)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Генерация"):
        claim = str(row["Headline1"]).strip()
        label = int(row["label"])

        e_cs = call_ollama(CS_PROMPT.format(claim=claim))

        e_tx = call_ollama(TX_PROMPT.format(claim=claim))

        rows.append({
            "claim"                  : claim,
            "commonsense_explanation": e_cs,
            "textual_explanation"    : e_tx,
            "label"                  : label,
        })

    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Done: {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stances", default="fakenews/train_stances.csv")
    p.add_argument("--bodies",  default="fakenews/train_bodies.csv")
    p.add_argument("--output",  default="final_dataset_for_advisor.csv")
    p.add_argument("--start",   type=int, default=None)
    p.add_argument("--end",     type=int, default=None)
    main(p.parse_args())