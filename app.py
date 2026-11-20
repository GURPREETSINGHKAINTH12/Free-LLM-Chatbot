"""
Chatbot Backend - FastAPI + Free LLM (Ollama or Hugging Face)

Features:
- Local LLM via Ollama (default, free, no API key) or Hugging Face Router API
- Persistent chat history stored in SQLite (survives server restarts)
- File upload & lightweight RAG (chat with uploaded TXT, MD, PDF, CSV, JSON, code files)
"""

import io
import math
import os
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from typing import Iterator, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Set USE_HF=true and HF_TOKEN=your_token to use Hugging Face instead of Ollama
USE_HUGGINGFACE = os.getenv("USE_HF", "false").lower() in ("true", "1", "yes")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Ollama settings (used when USE_HF is false)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# How many recent messages to send to the LLM for context (all messages are persisted in SQLite)
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

# Default session when none provided
DEFAULT_SESSION = "default"

# Upload / RAG settings
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RAG_TOP_K = 4

SUPPORTED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".csv",
    ".html", ".css", ".xml", ".yaml", ".yml", ".log", ".ini", ".cfg",
    ".sh", ".bat", ".sql", ".java", ".c", ".cpp", ".h", ".go", ".rs",
}

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(_BASE_DIR, "chat_history.db"))

app = FastAPI(title="Free LLM Chatbot", version="1.1.0")
template_env = Environment(loader=FileSystemLoader(os.path.join(_BASE_DIR, "templates")))


# ---------------------------------------------------------------------------
# SQLite Persistence (Chat History + Uploaded Documents)
# ---------------------------------------------------------------------------
@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()


init_db()


def load_history(session_id: str, limit: Optional[int] = None) -> list[dict]:
    """Load persisted messages for a session in chronological order."""
    with get_db() as conn:
        if limit is not None:
            rows = conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def save_message(session_id: str, role: str, content: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()


def clear_history(session_id: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Document Extraction, Chunking & Lightweight RAG Retrieval
# ---------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with", "this", "what", "which", "who", "how",
    "can", "you", "me", "my", "about", "please", "tell",
}


def tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 1 and tok not in STOP_WORDS
    ]


def extract_text_from_upload(filename: str, raw_bytes: bytes) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PDF support requires 'pypdf'. Install it with: pip install pypdf",
            )
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(p.strip() for p in pages if p.strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")
    elif ext in SUPPORTED_TEXT_EXTENSIONS:
        text = raw_bytes.decode("utf-8", errors="replace").strip()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Upload a PDF or plain-text/code file.",
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file contains no readable text.")
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    cleaned = re.sub(r"\r\n?", "\n", text).strip()
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += step
    return chunks


def save_document(session_id: str, filename: str, chunks: list[str]) -> dict:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO documents (session_id, filename) VALUES (?, ?)",
            (session_id, filename),
        )
        doc_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO document_chunks (document_id, chunk_index, content) VALUES (?, ?, ?)",
            [(doc_id, idx, chunk) for idx, chunk in enumerate(chunks)],
        )
        conn.commit()
    return {"id": doc_id, "filename": filename, "chunks": len(chunks)}


def list_documents(session_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.filename, COUNT(c.id) AS chunks
            FROM documents d
            LEFT JOIN document_chunks c ON c.document_id = d.id
            WHERE d.session_id = ?
            GROUP BY d.id
            ORDER BY d.id ASC
            """,
            (session_id,),
        ).fetchall()
    return [{"id": r["id"], "filename": r["filename"], "chunks": r["chunks"]} for r in rows]


def delete_document(session_id: str, doc_id: int) -> bool:
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE id = ? AND session_id = ?",
            (doc_id, session_id),
        )
        conn.commit()
        return cur.rowcount > 0


def retrieve_relevant_chunks(session_id: str, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
    """Retrieve the top-K most relevant document chunks for the query using TF-IDF scoring."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT d.filename, c.chunk_index, c.content
            FROM document_chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.session_id = ?
            ORDER BY d.id ASC, c.chunk_index ASC
            """,
            (session_id,),
        ).fetchall()

    if not rows:
        return []

    query_tokens = set(tokenize(query))
    chunk_token_counts = [Counter(tokenize(r["content"])) for r in rows]
    num_chunks = len(rows)

    # Document frequency for each query token
    df = {
        tok: sum(1 for counts in chunk_token_counts if counts[tok] > 0)
        for tok in query_tokens
    }

    scored = []
    for row, counts in zip(rows, chunk_token_counts):
        score = 0.0
        total_terms = sum(counts.values()) or 1
        for tok in query_tokens:
            tf = counts[tok] / total_terms
            if tf > 0:
                idf = math.log((num_chunks + 1) / (df[tok] + 1)) + 1.0
                score += tf * idf
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)

    # If the user asks a generic question (e.g. "summarize this file"), fall back to first chunks
    if scored[0][0] == 0.0:
        selected = rows[:top_k]
    else:
        selected = [row for score, row in scored[:top_k] if score > 0.0]

    return [
        {"filename": r["filename"], "chunk_index": r["chunk_index"], "content": r["content"]}
        for r in selected
    ]


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    message: str = Field(..., min_length=1, description="User message")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    history: Optional[list[dict]] = None


class ChatResponse(BaseModel):
    """Response sent back to the client."""

    reply: str
    model_used: str
    history: list[dict]
    sources: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt templating: system + optional RAG context + history + user message
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. Answer concisely and clearly. "
    "If you don't know something, say so."
)


def build_messages(
    history: list[dict],
    new_user_message: str,
    rag_chunks: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Build messages list for LLM: system prompt (+ optional RAG context) + recent turns + user message.
    Same structure works for both Ollama /api/chat and Hugging Face chat API.
    """
    system_content = SYSTEM_PROMPT
    if rag_chunks:
        context_blocks = [
            f"[Source: {c['filename']}]\n{c['content']}" for c in rag_chunks
        ]
        system_content += (
            "\n\nUse the following excerpts from the user's uploaded documents when relevant "
            "to answer their question. Cite the source filename when referencing document content.\n\n"
            + "\n\n---\n\n".join(context_blocks)
        )

    messages = [{"role": "system", "content": system_content}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": new_user_message})
    return messages


# ---------------------------------------------------------------------------
# LLM calls: Ollama (local) vs Hugging Face (Router API)
# ---------------------------------------------------------------------------
async def call_ollama(messages: list[dict], temperature: float) -> str:
    """
    Call local Ollama /api/chat. No API key needed.
    Ensure Ollama is running and you have pulled a model: ollama run mistral
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message", {}) or {}).get("content", "").strip()


async def call_huggingface(messages: list[dict], temperature: float) -> str:
    """
    Call Hugging Face Inference Providers Router API (OpenAI-compatible).
    Requires HF_TOKEN in env.
    """
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Hugging Face is enabled but HF_TOKEN is not set. Add HF_TOKEN to your environment.",
        )
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "model": HF_MODEL,
                "messages": messages,
                "max_tokens": 512,
                "temperature": temperature,
            },
        )
        if resp.status_code != 200:
            detail = resp.text or resp.reason_phrase
            raise HTTPException(status_code=resp.status_code, detail=detail)
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise HTTPException(status_code=502, detail="Empty response from Hugging Face")
        return (choices[0].get("message", {}) or {}).get("content", "").strip()


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat UI."""
    template = template_env.get_template("index.html")
    return template.render()


@app.get("/health")
async def health():
    """Health check; optionally verify Ollama is reachable when not using HF."""
    if not USE_HUGGINGFACE:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if r.status_code != 200:
                    return {"status": "degraded", "ollama": "unreachable"}
                return {"status": "ok", "backend": "ollama", "model": OLLAMA_MODEL}
        except Exception as e:
            return {"status": "degraded", "ollama": str(e)}
    return {"status": "ok", "backend": "huggingface", "model": HF_MODEL}


@app.get("/api/history")
async def get_chat_history(session_id: str = DEFAULT_SESSION):
    """Return full persisted chat history for a session."""
    return {"history": load_history(session_id)}


@app.delete("/api/history")
async def delete_chat_history(session_id: str = DEFAULT_SESSION):
    """Clear persisted chat history for a session."""
    clear_history(session_id)
    return {"status": "cleared"}


@app.get("/api/documents")
async def get_documents(session_id: str = DEFAULT_SESSION):
    """List uploaded documents for a session."""
    return {"documents": list_documents(session_id)}


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = DEFAULT_SESSION,
):
    """Upload a text/code/PDF file and index it for RAG chat."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File is too large (max 5 MB).")

    text = extract_text_from_upload(file.filename, raw_bytes)
    chunks = chunk_text(text)
    doc = save_document(session_id, file.filename, chunks)
    return {"document": doc}


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: int, session_id: str = DEFAULT_SESSION):
    """Remove an uploaded document."""
    deleted = delete_document(session_id, doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "deleted", "id": doc_id}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session_id: str = DEFAULT_SESSION):
    """
    Main chat endpoint.
    - Loads recent persisted history from SQLite and retrieves relevant document chunks (RAG).
    - Calls Ollama or Hugging Face, then saves the new turn to SQLite.
    - Returns the reply, model used, updated history, and cited document sources.
    """
    user_message = request.message.strip()
    temperature = request.temperature

    recent_history = load_history(session_id, limit=MAX_HISTORY_MESSAGES)
    rag_chunks = retrieve_relevant_chunks(session_id, user_message)
    sources = list(dict.fromkeys(c["filename"] for c in rag_chunks))

    messages = build_messages(recent_history, user_message, rag_chunks=rag_chunks)
    if USE_HUGGINGFACE:
        reply = await call_huggingface(messages, temperature)
        model_used = HF_MODEL
    else:
        reply = await call_ollama(messages, temperature)
        model_used = OLLAMA_MODEL

    save_message(session_id, "user", user_message)
    save_message(session_id, "assistant", reply)

    return ChatResponse(
        reply=reply,
        model_used=model_used,
        history=load_history(session_id),
        sources=sources,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
