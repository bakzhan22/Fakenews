import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from colorama import Fore, init
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

init(autoreset=True)
load_dotenv()



def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def text_id(value: str, size: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:size]


def _parse_retry_delay(error_str: str, default: float = 65.0) -> float:
    """
    Extract retryDelay seconds from a Google API error string.

    Handles formats like:
      'retryDelay': '30s'
      retryDelay: 30s
      retry after 45
    Returns the parsed value + 5s buffer, or `default` if nothing found.
    """
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+\.?\d*)s?['\"]?", error_str)
    if match:
        return float(match.group(1)) + 5

    match2 = re.search(r"retry\s+after\s+(\d+)", error_str, re.IGNORECASE)
    if match2:
        return float(match2.group(1)) + 5

    return default

class ChromaDBManager:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

        # Detect GPU automatically; fall back to CPU if unavailable
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        print(Fore.BLUE + f"Loading embedding model '{model_name}' on {device}...")

        from langchain_huggingface import HuggingFaceEmbeddings
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 128},
        )
        print(Fore.GREEN + "Embedding model ready.")

    def create_or_load_db(self, collection_name: str = "gov_news_gemini") -> Chroma:
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )



class DocumentProcessor:
    def __init__(self):
        chunk_size    = int(os.getenv("CHUNK_SIZE", "1100"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "180"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ", ", " ", ""],
            length_function=len,
        )

    @staticmethod
    def _norm_col_name(name: str) -> str:
        return "".join(ch for ch in str(name).strip().lower() if ch.isalnum() or ch == "_")

    def _pick_columns(self, df: pd.DataFrame) -> Tuple[str, Optional[str]]:
        text_candidates = {"text", "content", "body", "article", "news", "txt", "текст"}
        id_candidates   = {"id", "new_id", "doc_id", "document_id", "news_id"}
        normalized = {self._norm_col_name(col): str(col) for col in df.columns}
        text_col, id_col = None, None

        for cand in text_candidates:
            if self._norm_col_name(cand) in normalized:
                text_col = normalized[self._norm_col_name(cand)]
                break
        if text_col is None:
            for norm_name, original in normalized.items():
                if "text" in norm_name or "текст" in norm_name:
                    text_col = original
                    break
        for cand in id_candidates:
            if self._norm_col_name(cand) in normalized:
                id_col = normalized[self._norm_col_name(cand)]
                break
        if text_col is None:
            raise ValueError(f"Could not find text column. Available: {list(df.columns)}")
        return text_col, id_col

    def _table_to_documents(self, df: pd.DataFrame, source_path: Path) -> List[Document]:
        text_col, id_col = self._pick_columns(df)
        docs = []
        source_hash = file_sha1(source_path)
        for idx, row in df.iterrows():
            raw_text = row.get(text_col)
            if pd.isna(raw_text):
                continue
            content = str(raw_text).strip()
            if not content:
                continue
            raw_id = row.get(id_col) if id_col else idx
            doc_id = str(raw_id).strip() if str(raw_id).strip() else str(idx)
            docs.append(Document(page_content=content, metadata={
                "doc_id": doc_id,
                "source_file": str(source_path.name),
                "source_path": str(source_path),
                "source_type": source_path.suffix.lower().lstrip("."),
                "source_hash": source_hash,
                "row_number": int(idx),
            }))
        return docs

    def _load_excel(self, path: Path) -> List[Document]:
        return self._table_to_documents(pd.read_excel(path), path)

    def _load_csv(self, path: Path) -> List[Document]:
        return self._table_to_documents(pd.read_csv(path), path)

    def _load_txt(self, path: Path) -> List[Document]:
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            return []
        return [Document(page_content=content, metadata={
            "doc_id": path.stem,
            "source_file": str(path.name),
            "source_path": str(path),
            "source_type": "txt",
            "source_hash": file_sha1(path),
            "row_number": 0,
        })]

    def _load_pdf(self, path: Path) -> List[Document]:
        try:
            from pypdf import PdfReader
        except ImportError:
            print(Fore.YELLOW + "Install pypdf: pip install pypdf")
            return []
        reader  = PdfReader(str(path))
        pages   = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
        content = "\n\n".join(pages).strip()
        return [Document(page_content=content, metadata={
            "doc_id": path.stem,
            "source_file": str(path.name),
            "source_path": str(path),
            "source_type": "pdf",
            "source_hash": file_sha1(path),
            "row_number": 0,
        })] if content else []

    def load_documents(self, input_path: Path) -> List[Document]:
        files   = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"))
        allowed = {".xlsx", ".xls", ".csv", ".txt", ".pdf", ".docx"}
        loaded_docs: List[Document] = []
        for path in files:
            if path.suffix.lower() not in allowed:
                continue
            try:
                ext = path.suffix.lower()
                if ext in {".xlsx", ".xls"}:
                    docs = self._load_excel(path)
                elif ext == ".csv":
                    docs = self._load_csv(path)
                elif ext == ".txt":
                    docs = self._load_txt(path)
                elif ext == ".pdf":
                    docs = self._load_pdf(path)
                else:
                    docs = []
                loaded_docs.extend(docs)
                print(Fore.GREEN + f"Loaded {len(docs)} docs from {path.name}")
            except Exception as e:
                print(Fore.RED + f"Failed to load {path}: {e}")
        return loaded_docs

    def split_documents(self, documents: Sequence[Document]) -> Tuple[List[Document], List[str]]:
        splits: List[Document] = []
        ids:    List[str]      = []
        for doc in documents:
            base_doc_id = str(doc.metadata.get("doc_id", "unknown"))
            doc_splits  = self.text_splitter.split_documents([doc])
            for idx, chunk in enumerate(doc_splits):
                chunk_hash = text_id(chunk.page_content, size=10)
                chunk_id   = f"{base_doc_id}:{idx}:{chunk_hash}"
                chunk.metadata.update({"chunk_index": idx, "chunk_id": chunk_id})
                splits.append(chunk)
                ids.append(chunk_id)
        return splits, ids

    def index_documents(
        self,
        documents: Sequence[Document],
        ids: Sequence[str],
        vector_store: Chroma,
        checkpoint_path: Optional[Path] = None,
    ):
        """
        Index documents into ChromaDB with:
          - Separate retry logic for 429 (quota) vs 503 (service down)
          - Resume-from-checkpoint: saves progress to a JSON file so a crash
            can be continued without re-indexing already-completed batches.

        Checkpoint file is written after every successful batch.
        Delete it manually if you want to re-index from scratch.

        Tuneable via env vars:
          INDEX_BATCH_SIZE  (default 96)
          BATCH_SLEEP_SEC   (default 62)
          INDEX_MAX_RETRIES (default 10)
        """
        batch_size  = int(os.getenv("INDEX_BATCH_SIZE", "96"))
        batch_sleep = float(os.getenv("BATCH_SLEEP_SEC", "62"))
        max_retries = int(os.getenv("INDEX_MAX_RETRIES", "10"))

        total = len(documents)

        # ── Resume from checkpoint ────────────────────────────────────────────
        start_batch = 0
        if checkpoint_path and checkpoint_path.exists():
            try:
                ckpt = json.loads(checkpoint_path.read_text())
                start_batch = int(ckpt.get("last_completed_batch", 0))
                #print(Fore.CYAN + f"[Checkpoint] Resuming from batch {start_batch + 1} "
                 #                 f"(skipping {start_batch * batch_size} already-indexed chunks).")
            except Exception as e:
                print(Fore.YELLOW + f"[Checkpoint] Could not read checkpoint: {e}. Starting fresh.")
                start_batch = 0

        i         = start_batch * batch_size
        batch_num = start_batch

        while i < total:
            batch_docs = list(documents[i: i + batch_size])
            batch_ids  = list(ids[i: i + batch_size])
            batch_num += 1
            success    = False

            for attempt in range(1, max_retries + 1):
                try:
                    vector_store.add_documents(batch_docs, ids=batch_ids)
                    done = i + len(batch_docs)
                    print(Fore.GREEN + f"Batch {batch_num}: {done}/{total} ({done / total * 100:.1f}%)")
                    success = True

                    # Save checkpoint after every successful batch
                    if checkpoint_path:
                        checkpoint_path.write_text(
                            json.dumps({"last_completed_batch": batch_num, "total": total})
                        )
                    break

                except Exception as e:
                    err_str = str(e)

                    # ── 429 / quota exhausted ─────────────────────────────────
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        if attempt == 1:
                            print(Fore.RED + f"[DEBUG] Rate-limit error: {err_str[:500]}")
                        suggested = _parse_retry_delay(err_str, default=0.0)
                        # Exponential backoff: 30 → 60 → 120 → 240 → 300s cap
                        wait = suggested if suggested > 0 else min(30 * (2 ** (attempt - 1)), 300)
                        print(Fore.YELLOW + f"[429] Rate limited — attempt {attempt}/{max_retries}. "
                                            f"Sleeping {wait:.0f}s...")
                        time.sleep(wait)

                    # ── 503 / service temporarily unavailable ─────────────────
                    elif "503" in err_str or "UNAVAILABLE" in err_str:
                        if attempt == 1:
                            print(Fore.RED + f"[DEBUG] Service unavailable: {err_str[:500]}")
                        # Linear backoff: 60 → 120 → 180 → ... → 600s cap
                        wait = min(60 * attempt, 600)
                        print(Fore.YELLOW + f"[503] Service unavailable — attempt {attempt}/{max_retries}. "
                                            f"Sleeping {wait:.0f}s...")
                        time.sleep(wait)

                    # ── Any other error ───────────────────────────────────────
                    else:
                        print(Fore.RED + f"Batch {batch_num} error (attempt {attempt}): {e}")
                        if attempt == max_retries:
                            raise
                        time.sleep(5)

            if not success:
                raise RuntimeError(
                    f"Failed to index batch {batch_num} after {max_retries} attempts.\n"
                    f"Progress saved. Re-run the script to resume from batch {batch_num}."
                )

            i += batch_size
            if i < total:
                print(Fore.CYAN + f"Sleeping {batch_sleep:.0f}s between batches...")
                time.sleep(batch_sleep)

        # All done — remove checkpoint
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()
            print(Fore.GREEN + "[Checkpoint] Indexing complete. Checkpoint removed.")

class TerminalRAG:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store

        # ── Config from .env ──────────────────────────────────────────────────
        self.per_query_k               = int(os.getenv("PER_QUERY_K", "8"))
        self.final_k                   = int(os.getenv("FINAL_K", "10"))
        self.score_threshold           = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "1.55"))
        self.keyword_k                 = int(os.getenv("KEYWORD_K", "20"))
        self.min_evidence_chunks       = int(os.getenv("MIN_EVIDENCE_CHUNKS", "2"))
        self.min_reciprocal_rank_score = float(os.getenv("MIN_RRF_SCORE", "0.028"))
        self.max_context_evidence      = int(os.getenv("MAX_CONTEXT_EVIDENCE", "8"))
        self.enable_false_category     = os.getenv("ENABLE_FALSE_CATEGORY", "true").lower() == "true"
        self.debug_retrieve_k          = int(os.getenv("DEBUG_RETRIEVE_K", "30"))
        self.debug_log_n               = int(os.getenv("DEBUG_LOG_N", "30"))

        # ── LLM ───────────────────────────────────────────────────────────────
        self.llm = OllamaLLM(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b"),
            temperature=0.1,
        )

        # ── Prompts ───────────────────────────────────────────────────────────
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a fact-grounded assistant for government news.\n"
                "Use only the provided context. Cite doc_id/chunk inline.\n"
                "CRITICAL: You MUST respond ONLY in Russian (на русском языке). "
                "Never respond in Chinese, English, or any other language.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Ответ (только на русском языке):"
            ),
        )

        # ── Collection cache (lazy) ────────────────────────────────────────────
        self._collection_cache: Optional[Dict] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_collection_cache(self):
        if self._collection_cache is None:
            print(Fore.CYAN + "[Cache] Loading full collection for keyword search…")
            self._collection_cache = self.vector_store.get()
            n = len((self._collection_cache.get("ids") or []))
            print(Fore.CYAN + f"[Cache] Loaded {n} chunks.")

    def invalidate_cache(self):
        self._collection_cache = None

    def _doc_key(self, doc: Document) -> str:
        return str(doc.metadata.get("chunk_id") or text_id(doc.page_content))

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())
        # Add bigrams so "кевин миллс" matches as a unit
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _vector_search(self, query: str) -> Tuple[List[Document], Optional[float]]:
        pairs = self.vector_store.similarity_search_with_score(query, k=self.per_query_k)
        if not pairs:
            return [], None
        ranked     = [doc for doc, _ in sorted(pairs, key=lambda x: x[1])]
        best_score = float(pairs[0][1])
        return ranked, best_score

    def _keyword_search(self, query: str) -> List[Document]:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []
        try:
            self._ensure_collection_cache()
            data      = self._collection_cache
            all_texts = data.get("documents") or []
            all_metas = data.get("metadatas") or []

            scored: List[Tuple[int, Document]] = []
            for text, meta in zip(all_texts, all_metas):
                if not text:
                    continue
                doc_tokens = set(self._tokenize(text))
                overlap    = len(query_tokens & doc_tokens)
                if overlap > 0:
                    scored.append((overlap, Document(page_content=text, metadata=meta or {})))

            scored.sort(key=lambda x: x[0], reverse=True)

            if self.debug_log_n > 0 and scored:
                for sc, doc in scored[:3]:
                    snippet = doc.page_content[:120].replace("\n", " ")

            return [doc for _, doc in scored[:self.keyword_k]]

        except Exception as e:
            return []

    def retrieve_hybrid(self, query: str) -> Tuple[List[Document], Optional[float], float]:
        vector_ranked, best_vec_score = self._vector_search(query)
        keyword_hits                  = self._keyword_search(query)

        rrf_scores: Dict[str, float]    = {}
        seen_docs:  Dict[str, Document] = {}

        def add_rrf(doc: Document, rank: int):
            key = self._doc_key(doc)
            seen_docs[key]  = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)

        for rank, doc in enumerate(vector_ranked):
            add_rrf(doc, rank)
        for rank, doc in enumerate(keyword_hits):
            add_rrf(doc, rank)

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        selected    = [seen_docs[k] for k in sorted_keys[:self.final_k]]
        top_rrf     = rrf_scores[sorted_keys[0]] if sorted_keys else 0.0

        if self.debug_log_n > 0:
            #print(Fore.CYAN + f"\n[Retrieve] query='{query}'")
            #print(Fore.CYAN + f"  vector hits: {len(vector_ranked)}  keyword hits: {len(keyword_hits)}")
            #print(Fore.CYAN + f"  top RRF: {top_rrf:.4f}  best vec score: {best_vec_score}")
            for i, k in enumerate(sorted_keys[:min(self.debug_log_n, 5)]):
                snippet = seen_docs[k].page_content[:100].replace("\n", " ")
                #print(Fore.CYAN + f"  [{i+1}] rrf={rrf_scores[k]:.4f}  {snippet}…")

        return selected, best_vec_score, top_rrf

    # ── Answer generation ─────────────────────────────────────────────────────

    def answer_question(self, query: str) -> str:
        docs, best_vec_score, top_rrf = self.retrieve_hybrid(query)

        if not docs:
            return "Информация не найдена в базе данных."

        #if top_rrf < self.min_reciprocal_rank_score:
            #print(Fore.YELLOW + f"[Warn] Top RRF {top_rrf:.4f} below threshold "
             #                   f"{self.min_reciprocal_rank_score} — results may be weak.")

        context_docs = docs[:self.max_context_evidence]
        context = "\n\n".join(
            f"[Doc {i} | chunk_id={d.metadata.get('chunk_id', '?')}]:\n{d.page_content}"
            for i, d in enumerate(context_docs, 1)
        )

        return self.llm.invoke(self.qa_prompt.format(context=context, question=query))

    # ── REPL ──────────────────────────────────────────────────────────────────

    def run(self):
        #print(Fore.GREEN + "RAG Assistant Ready. Type 'exit' to quit.")
        while True:
            try:
                query = input(Fore.CYAN + "\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break
            print(Fore.YELLOW + "\nAssistant:\n" + self.answer_question(query))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    current_dir      = Path(__file__).parent
    chroma_directory = current_dir / "chromadb_gemini"
    checkpoint_file  = current_dir / ".index_checkpoint.json"

    if not os.getenv("GOOGLE_API_KEY"):
        print(Fore.RED + "Error: GOOGLE_API_KEY not set in .env")
        return

    db_manager   = ChromaDBManager(str(chroma_directory))
    vector_store = db_manager.create_or_load_db(
        collection_name=os.getenv("CHROMA_COLLECTION", "gov_news_gemini")
    )

    try:
        collection_size = len(vector_store.get().get("ids", []))
    except Exception:
        collection_size = 0

    needs_indexing = collection_size == 0 or checkpoint_file.exists()

    if needs_indexing:
        processor   = DocumentProcessor()
        input_path  = current_dir / "data"
        docs        = processor.load_documents(input_path)
        splits, ids = processor.split_documents(docs)
        processor.index_documents(splits, ids, vector_store, checkpoint_path=checkpoint_file)

    rag = TerminalRAG(vector_store)
    rag.invalidate_cache()
    rag.run()


if __name__ == "__main__":
    main()
