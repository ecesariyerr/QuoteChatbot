import os
import re
import json
from typing import List, Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========= FastAPI =========
app = FastAPI(title="Erciyes RAG API", version="1.0", openapi_url="/openapi.json")

# CORS (tarayıcıdan test / Open WebUI kullanımı için açık bırakıyoruz)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Ortam Değişkenleri =========
DOCS_DIR      = os.getenv("DOCS_DIR", "/data/texts")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_DEFAULT = int(os.getenv("TOP_K", "4"))
LM_BASE       = os.getenv("LMSTUDIO_BASE", "http://host.docker.internal:8081/v1")
MODEL_NAME    = "openhermes-2.5-mistral-7b"
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "512"))

# ========= Global Bellek =========
_docs_raw: List[Dict[str, Any]] = []
_chunks: List[Dict[str, Any]] = []
_vectorizer: Optional[TfidfVectorizer] = None
_matrix = None


# ===================== Yardımcı Fonksiyonlar =====================

def _clean_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip().strip('\'"').replace('\\', '')
    return re.sub(r'[",;\s]+$', '', u)

def _extract_meta(text: str, fname: str):
    title, url = None, None
    m_title = re.search(r"^TITLE:\s*(.+)$", text, flags=re.I | re.M)
    m_url   = re.search(r"^URL:\s*(.+)$",   text, flags=re.I | re.M)
    if m_title:
        title = m_title.group(1).strip()
    if m_url:
        url = _clean_url(m_url.group(1))
    if not title:
        title = os.path.splitext(os.path.basename(fname))[0]
    return title, url

def _chunk_text(s: str, size: int, overlap: int) -> List[str]:
    s = s.strip()
    chunks, i, n = [], 0, len(s)
    while i < n:
        j = min(i + size, n)
        chunks.append(s[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def _build_index() -> Dict[str, int]:
    global _docs_raw, _chunks, _vectorizer, _matrix
    _docs_raw, _chunks = [], []

    for root, _, files in os.walk(DOCS_DIR):
        for fn in files:
            if not fn.lower().endswith(".txt"):
                continue
            path = os.path.join(root, fn)
            try:
                raw = open(path, "r", encoding="utf-8").read()
                title, url = _extract_meta(raw, fn)

                # JSON formatındaysa sadece content al
                try:
                    obj = json.loads(raw)
                    body = obj.get("content", "").strip()
                    title = obj.get("title", title)
                    url   = obj.get("url", url)
                except json.JSONDecodeError:
                    m = re.search(r"^CONTENT:\s*(.*)$", raw, flags=re.I | re.S | re.M)
                    body = m.group(1).strip() if m else raw.strip()

                _docs_raw.append({
                    "file": fn,
                    "title": title,
                    "url": url,
                    "content": body
                })
            except Exception as e:
                print("read_error", path, e)

    for d in _docs_raw:
        parts = _chunk_text(d["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, p in enumerate(parts):
            _chunks.append({
                "text": p,
                "file": d["file"],
                "title": d["title"],
                "url": d["url"],
                "idx": idx
            })

    texts = [c["text"] for c in _chunks] if _chunks else ["empty"]
    _vectorizer = TfidfVectorizer()
    _matrix = _vectorizer.fit_transform(texts)

    return {"files": len(_docs_raw), "chunks": len(_chunks)}

def _search(q: str, top_k: int) -> List[Dict[str, Any]]:
    if not _chunks:
        return []
    qv   = _vectorizer.transform([q])
    sims = cosine_similarity(qv, _matrix).flatten()
    order = sims.argsort()[::-1][:top_k]
    out: List[Dict[str, Any]] = []
    for i in order:
        c = _chunks[i]
        out.append({
            "score": float(sims[i]),
            "file": c["file"],
            "title": c["title"],
            "url": c["url"],
            "chunk_index": c["idx"],
            "text": c["text"],
        })
    return out

def _build_prompt(query: str, hits: List[Dict[str, Any]]):
    # Kaynak listesini hazırla
    seen = set()
    sources: List[str] = []
    for h in hits:
        key = h.get("url") or h["file"]
        if key in seen:
            continue
        seen.add(key)
        if h.get("url"):
            sources.append(f"- {h['title']} → {h['url']}")
        else:
            sources.append(f"- {h['title']}")

    context = "\n\n".join(h["text"] for h in hits)
    system_p = (
        # 1) Tell the model when and how to call the tool:
        "You are a helpful AI assistant. When you need to look up information in the quotes dataset, "
        "emit exactly this JSON and nothing else:\n"
        "{\n"
        "  \"tool_calls\": [\n"
        "    {\"name\": \"rag_answer_rag_answer_post\", \"parameters\": {\"query\": \"<USER_QUERY>\"}}\n"
        "  ]\n"
        "}\n\n"
        # 2) After the tool is run, OpenWebUI will inject its response back here.
        #    Now you should answer the user's original question using that tool output.
        "After the tool call completes, you will receive the tool’s JSON response. "
        "Then, use the contents of that tool response to write a concise English answer. "
        "Do not repeat the question or context. "
        "If the tool returns no useful information, say “I’m sorry, I don’t have enough information.” "
        "At the end, include “Sources:” followed by the list of sources."
    )

    user_p = (
            f"Question: {query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            # 3) Give a reminder of what you want from the tool:
            "Please list verbatim the tags associated with this quote as they appear in the source metadata.\n\n"
            # 4) Finally remind how the final answer should look:
            "Once you have the tool response, write only the answer—do not repeat the question or context—and then:\n"
            "Finally, append sources in this exact format:\n"
            + "\n".join(sources)
    )
    return system_p, user_p


# ===================== API Uçları =====================

@app.get("/health", include_in_schema=False)
def health(query: Optional[str] = Query(None, description="Sorgu (opsiyonel)")):
    if query:
        hits = _search(query, TOP_K_DEFAULT)
        sys_p, user_p = _build_prompt(query, hits)
        try:
            r = requests.post(
                f"{LM_BASE}/chat/completions",
                headers={
                    "Authorization": "Bearer lm-studio",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": sys_p},
                        {"role": "user",   "content": user_p},
                    ],
                    "temperature": 0.2,
                    "max_tokens": MAX_TOKENS,
                    "stream": False,
                },
                timeout=120
            )
            r.raise_for_status()
            data = r.json()
            return {
                "answer": data["choices"][0]["message"]["content"],
                "hits": hits
            }
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LM call failed: {e}")

    return {
        "status": "ok",
        "docs_dir": DOCS_DIR,
        "files": len(_docs_raw),
        "chunks": len(_chunks),
        "model": MODEL_NAME,
        "lm_base": LM_BASE,
    }

@app.post("/reload", include_in_schema=False)
def reload_index():
    stats = _build_index()
    return {"reindexed": True, **stats}

class QueryBody(BaseModel):
    query: str
    top_k: Optional[int] = None

@app.post("/query", include_in_schema=False)
def query(body: QueryBody):
    top_k = body.top_k or TOP_K_DEFAULT
    return {"query": body.query, "top_k": top_k, "results": _search(body.query, top_k)}

class AnswerBody(BaseModel):
    query: str
    top_k: Optional[int] = None
    temperature: Optional[float] = 0.2

@app.post("/answer", include_in_schema=False)
def answer(body: AnswerBody):
    top_k = body.top_k or TOP_K_DEFAULT
    hits = _search(body.query, top_k)
    sys_p, user_p = _build_prompt(body.query, hits)
    try:
        r = requests.post(
            f"{LM_BASE}/chat/completions",
            headers={
                "Authorization": "Bearer lm-studio",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": sys_p},
                    {"role": "user",   "content": user_p},
                ],
                "temperature": body.temperature or 0.2,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            },
            timeout=120
        )
        r.raise_for_status()
        data = r.json()
        return {"answer": data["choices"][0]["message"]["content"], "hits": hits}
    except Exception as e:
        return {
            "error": f"LM call failed: {e}",
            "hits": hits,
            "constructed_prompt": {"system": sys_p, "user": user_p},
        }

class RAGQuery(BaseModel):
    query: str

@app.post("/rag-answer")
def rag_answer(body: RAGQuery):
    try:
        resp = requests.post(
            "http://localhost:7000/answer",
            json={"query": body.query},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"RAG API hatası: {e}")

# ===================== Yaşam Döngüsü =====================

@app.on_event("startup")
def on_startup():
    print("Indexing…")
    stats = _build_index()
    print("Ready:", stats)
