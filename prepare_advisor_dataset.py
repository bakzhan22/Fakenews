import re
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

OLLAMA_URL = "http://localhost:11434/api/generate"

CS_PROMPT = """Ты — система анализа новостей. Оцени логическую последовательность заголовка, используя только здравый смысл.
Не выноси вердикт — только анализируй логику и правдоподобность.
Заголовок: {claim}
Вывод (2-3 предложения, русский язык):"""

TX_PROMPT = """Ты — система анализа текста. Проанализируй стилистику заголовка новости.
Не выноси вердикт — только описывай стиль: нейтральный/сенсационный, наличие эмоций, кликбейта.
Заголовок: {claim}
Вывод (2-3 предложения, русский язык):"""

DB_PROMPT = """Ты — система проверки фактов. Используй свои знания для анализа утверждения.
Не выноси вердикт — только описывай известные факты по теме.
Заголовок: {claim}
Фактическое обоснование (2-3 предложения, русский язык):"""


def call_ollama(prompt: str, model: str = "gpt-oss:120b") -> str:
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model, "prompt": prompt, "stream": False
        }, timeout=120)
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"Ошибка: {e}"


def generate_from_tsv(tsv_path: str, model: str, limit: int = None) -> pd.DataFrame:
    """Генерирует все три объяснения из TSV файла."""
    df = pd.read_csv(tsv_path, sep="\t")
    if limit:
        df = df.head(limit)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Генерация ({tsv_path})"):
        claim = str(row["title"]).strip()
        label = int(row["is_fake"])

        e_cs = call_ollama(CS_PROMPT.format(claim=claim), model)
        e_tx = call_ollama(TX_PROMPT.format(claim=claim), model)
        e_db = call_ollama(DB_PROMPT.format(claim=claim), model)

        rows.append({
            "claim"                  : claim,
            "evidence_explanation"   : e_db,
            "commonsense_explanation": e_cs,
            "textual_explanation"    : e_tx,
            "label"                  : label,
        })

    return pd.DataFrame(rows)


def merge_and_clean(df_a: pd.DataFrame, df_b: pd.DataFrame = None) -> pd.DataFrame:
    VERDICT = re.compile(
        r"Вердикт\s*:\s*(TRUE|FALSE|UNVERIFIABLE)\.?\s*", re.IGNORECASE
    )
    LABEL_TOKENS = re.compile(r"\b(TRUE|FALSE|UNVERIFIABLE)\b", re.IGNORECASE)
    STARRED = re.compile(
        r"\*\*(real|fake|реальн\w*|фейк\w*|TRUE|FALSE)\*\*", re.IGNORECASE
    )

    def clean_ev(t):
        if not isinstance(t, str): return ""
        t = VERDICT.sub("", t)
        t = LABEL_TOKENS.sub("", t)
        t = STARRED.sub("", t)
        return t.strip()

    df = pd.DataFrame({
        "claim"                  : df_a["claim"],
        "evidence_explanation"   : (df_b["evidence_explanation"] if df_b is not None
                                    else df_a["evidence_explanation"]).apply(clean_ev),
        "commonsense_explanation": df_a["commonsense_explanation"],
        "textual_explanation"    : df_a["textual_explanation"],
        "label"                  : df_a["label"],
    })

    df = df.dropna(subset=["claim", "evidence_explanation", "label"])
    df = df[df["evidence_explanation"].str.len() > 10]
    df["commonsense_explanation"] = df["commonsense_explanation"].fillna("")
    df["textual_explanation"]     = df["textual_explanation"].fillna("")
    return df.reset_index(drop=True)


def check_leakage(df: pd.DataFrame):
    print("\nПроверка leakage:")
    cols = ["evidence_explanation", "commonsense_explanation", "textual_explanation"]
    critical = ["TRUE", "FALSE", "Вердикт", "is_fake", "fake", "real"]
    found = False
    for col in cols:
        if col not in df.columns: continue
        for kw in critical:
            mask = df[col].str.contains(kw, case=False, na=False)
            hits = mask.sum()
            if hits == 0: continue
            ld = df[mask]["label"].value_counts().to_dict()
            is_leak = len(ld) == 1 and hits > 20
            marker = "[LEAK]" if is_leak else "[ok]  "
            if is_leak:
                print(f"  {marker} '{kw}' в {col}: {hits} → {ld}")
                found = True
    if not found:
        print("  Критических утечек не обнаружено!")

    # TF-IDF тест сложности задачи
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score

    vec = TfidfVectorizer(max_features=3000)
    X = vec.fit_transform(df["claim"])
    y = df["label"]
    scores = cross_val_score(LogisticRegression(max_iter=500), X, y,
                             cv=3, scoring="f1_macro")
    print(f"\n  TF-IDF на claim: F1={scores.mean():.4f} ± {scores.std():.4f}")
    if scores.mean() > 0.95:
        print("  [!] Задача слишком простая — проверьте датасет")
    else:
        print("  [OK] Задача нетривиальна — Advisor Model имеет смысл")


def main(args):
    if args.generate:
        print(f"\nГенерация объяснений (модуль A) из {args.train}...")
        df_a = generate_from_tsv(args.train, args.model, args.limit)
        df_a.to_csv("final_dataset_for_advisor.csv", index=False, encoding="utf-8-sig")
        print("Сохранено: final_dataset_for_advisor.csv")
        df_b = None

    else:
        print(f"\nЗагрузка готовых файлов...")
        df_a = pd.read_csv(args.data_a)
        print(f"  Датасет A: {len(df_a)} записей")

        df_b = None
        if args.data_b:
            df_b = pd.read_csv(args.data_b)
            print(f"  Датасет B: {len(df_b)} записей")

    print("\nОбъединение и очистка...")
    df_final = merge_and_clean(df_a, df_b)

    print(f"\nИтоговый датасет: {len(df_final)} записей")
    print(df_final["label"].value_counts().to_string())

    check_leakage(df_final)

    df_final.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nСохранено: {args.output}")

    # Split info
    train_df, tmp = train_test_split(df_final, test_size=0.3,
                                     stratify=df_final["label"], random_state=42)
    val_df, test_df = train_test_split(tmp, test_size=0.5,
                                       stratify=tmp["label"], random_state=42)
    print(f"Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    print(f"""
Следующий шаг — обучение:
  python advisor_train.py \\
    --data_a {args.output} \\
    --data_b {args.output} \\
    --merge_strategy a_only \\
    --epochs 5
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--generate",  action="store_true",
                   help="Генерировать объяснения прямо сейчас")
    p.add_argument("--train",     default="fakenews_dataset/train.tsv")
    p.add_argument("--model",     default="gpt-oss:120b")
    p.add_argument("--limit",     type=int, default=None)
    p.add_argument("--data_a",    default="final_dataset_for_advisor.csv",
                   help="Готовый датасет модуля A")
    p.add_argument("--data_b",    default=None,
                   help="Готовый датасет модуля B (RAG), опционально")
    p.add_argument("--output",    default="advisor_dataset_final.csv")
    main(p.parse_args())
