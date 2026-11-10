# KazFactCheck — LLM-advisor pipeline

Code accompanying the paper

> **KazFactCheck: Benchmarking Evidence-Based Claim Verification in Kazakh and Russian Beyond Style and Source Shortcuts** (submitted to *Big Data and Cognitive Computing*, 2026).

This repository contains the **LLM-advisor pipeline** of the paper (Section 3.6, Appendices G and H): generation of three rationales for a claim with an open instruction-tuned LLM (Qwen3.8-27B via Ollama), hybrid retrieval over a local ChromaDB evidence base, and the advisor classifier (multilingual-e5-large with bidirectional cross-attention over the claim and the three rationales). The same code was used for the preliminary cross-attention experiment on the Russian-language "genre" corpus (Section 5.3, Appendix H).

The remaining parts of the reproducibility package — the KazFactCheck v1 manifest (claim identifiers, labels, groups and the temporal split), the cleaned fact-checker justifications and retrieved snippets with the full search log, the DÄLEL passage-level verifier and its aggregation variants, the stress tests and faithfulness scripts, and the predictions of every run — will be deposited on Zenodo upon acceptance of the paper. The DOI will be added here.

## Task

Binary claim verification: given a claim in Kazakh or Russian (and, optionally, evidence), predict **REAL** or **FAKE**. Labels come from Factcheck.kz verdicts; the pipeline itself never sees the verdict. Before classification, the LLM produces three rationales for each claim:

1. `evidence_explanation` — an evidence-based rationale over web pages / local documents retrieved by RAG;
2. `commonsense_explanation` — an analysis of the internal logic and plausibility of the claim;
3. `textual_explanation` — an analysis of style: sensationalism, clickbait, emotional pressure, capitalisation, exclamation marks.

Verdict-naming words are removed from the rationales before training (see `prepare_dalel_dataset.py`), and a TF-IDF probe estimates the residual label leakage.

## Repository structure

| File | Purpose |
| --- | --- |
| `bmwGpt2.py` | Indexes documents in ChromaDB and provides hybrid retrieval: dense search (multilingual-e5-large) + keyword search + Reciprocal Rank Fusion. Also an interactive query mode. |
| `generate_explanations.py` | Generates the common-sense and stylistic rationales. |
| `generate_rag_explanations.py` | Generates the evidence-based rationale with ChromaDB retrieval; falls back to the article body when the database is unavailable. |
| `prepare_dalel_dataset.py` | Generates or merges the rationales, removes verdict words, checks for label leakage and validates the dataset. |
| `dalel.py` | Trains the advisor classifier (`AdvisorModel`: shared encoder + bidirectional cross-attention) and evaluates it on the validation and test splits. |
| `test_predict.py` | Generates rationales for new claims, loads a trained checkpoint, predicts and computes metrics. |

## Requirements

- Python ≥ 3.10;
- [Ollama](https://ollama.com) with the LLM pulled locally (`qwen-3.8:27b` by default; set `OLLAMA_MODEL` or `--model` to use a smaller model);
- access to Hugging Face for the first download of `intfloat/multilingual-e5-large`;
- a CUDA GPU is recommended (all experiments in the paper were run on a single NVIDIA T4, 16 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy requests tqdm python-dotenv torch transformers scikit-learn colorama \
            chromadb langchain-chroma langchain-ollama langchain-core langchain-text-splitters \
            langchain-huggingface pypdf openpyxl
ollama pull qwen-3.8:27b && ollama serve
```

## Configuration

Create `.env` in the project root:

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen-3.8:27b
EMBEDDING_MODEL=intfloat/multilingual-e5-large
CHROMA_DIR=chromadb_gemini
CHROMA_COLLECTION=gov_news_gemini
GOOGLE_API_KEY=unused   # required by bmwGpt2.py at start-up; any non-empty value

CHUNK_SIZE=1100
CHUNK_OVERLAP=180
INDEX_BATCH_SIZE=96
BATCH_SLEEP_SEC=62
INDEX_MAX_RETRIES=10
PER_QUERY_K=8
FINAL_K=10
KEYWORD_K=20
MAX_CONTEXT_EVIDENCE=8
RETRIEVAL_SCORE_THRESHOLD=1.55
MIN_EVIDENCE_CHUNKS=2
MIN_RRF_SCORE=0.028
```

## Data formats

**Claims (TSV)** for `prepare_dalel_dataset.py --generate`: columns `title` (the claim text) and `is_fake` (numeric label). In the classifier, class `1` = REAL and class `0` = FAKE; invert `is_fake` if your file uses the opposite convention.

**Evidence base for RAG**: put documents into `data/`. Supported: `.csv`/`.xlsx` (text column named `text`, `content`, `body`, `article`, `news`, `txt` or `текст`; optional id column `id`, `new_id`, `doc_id`, `document_id`, `news_id`), `.txt`, and `.pdf` with a text layer. In the paper the base consisted of official Gov.kz and agency news.

**Final training CSV**:

```text
claim,evidence_explanation,commonsense_explanation,textual_explanation,label
```

## Running the pipeline

1. Build the evidence index (first run indexes `data/` into `chromadb_gemini/`; progress is checkpointed in `.index_checkpoint.json`):

   ```bash
   python bmwGpt2.py
   ```

2. Generate the rationales and assemble the dataset:

   ```bash
   python prepare_dalel_dataset.py --generate --train data/train.tsv \
          --model qwen-3.8:27b --output dalel_dataset_final.csv
   # trial run on 100 claims
   python prepare_dalel_dataset.py --generate --train data/train.tsv --limit 100 \
          --output dalel_dataset_sample.csv
   ```

   Alternatively, generate the rationales separately (`generate_explanations.py` → common-sense + stylistic, `generate_rag_explanations.py` → evidence) and merge the two CSV files with `prepare_dalel_dataset.py --data_a ... --data_b ... --output ...`. The files are merged by row position, so both must contain the same claims in the same order.

3. Train and evaluate the advisor classifier (seed 42, up to 15 epochs, best epoch by validation Macro F1):

   ```bash
   python dalel.py --data_a dalel_dataset_final.csv --model_name intfloat/multilingual-e5-large \
          --max_len 256 --epochs 15 --seed 42 --save_path advisor_model_best.pt
   ```

4. Predict on new claims:

   ```bash
   python test_predict.py --checkpoint advisor_model_best.pt --output test_predictions.csv
   ```

## Results reported in the paper

On the held-out test set of the Kazakh dataset with multi-level annotation (538 texts) the advisor reaches Macro F1 0.822 (95% CI [0.782; 0.860]) and 0.803 on its KazFactCheck part; on the Russian-language "genre" corpus it reaches 0.974 — see Appendix H (Tables A10–A11, Figure A2) for the protocol, confidence intervals and error analysis. Prompts and the pinned model revision are those used in the paper.

## Citation

```bibtex
@article{kazfactcheck2026,
  title   = {KazFactCheck: Benchmarking Evidence-Based Claim Verification in Kazakh and Russian Beyond Style and Source Shortcuts},
  author  = {Nekessova, Anargul and Lamasheva, Zhanar and Sambetbayeva, Madina and Nazymkhan, Aksaule and Mamatayeva, Dilnoza and Sidorova, Elena and Biibosunov, Bolotbek and Sabdenov, Aidos},
  journal = {Big Data and Cognitive Computing},
  year    = {2026},
  note    = {submitted}
}
```

## Funding and licence

This research was funded by the Science Committee of the Ministry of Science and Higher Education of the Republic of Kazakhstan, grant AP26195591 (2025–2027). Code is released under the MIT licence (see `LICENSE`); the data of the reproducibility package will be released under CC BY 4.0.
