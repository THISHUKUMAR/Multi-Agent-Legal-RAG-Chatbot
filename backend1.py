import os
import pdfplumber
import faiss
import numpy as np
import math
from typing import List, Dict, Any, Tuple

import google.generativeai as genai


# ---------------------------
#   GEMINI CONFIG
# ---------------------------

os.environ["GOOGLE_API_KEY"] = "AIzaSyCUIycD0goSuABem0Aungs95Lt_rkM6fa8"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# backend.py
# Backend for Multi-Agent Legal RAG Chatbot
# - Uses google.generativeai (Gemini) for embeddings + chat (with safe fallbacks)
# - Builds separate FAISS indices for Acts / Cases / Regulations
# - Returns retrieved chunks with file/page/chunk citations
# - Final synthesis uses Gemini to produce a concise answer with citations




# Model names (change if you have different model names)
EMBED_MODEL_NAME = "text-embedding-3-large"   # or "text-embedding-001" / "models/text-embedding-004"
CHAT_MODEL_NAME = "gemini-2.0-flash"          # or "gemini-2.5" etc.

# ---------------------------
# Utilities: PDF extraction
# ---------------------------
def extract_pages(file_obj) -> List[Tuple[str,int]]:
    """Return list of tuples (text, page_no) for the PDF file-like object."""
    pages = []
    try:
        with pdfplumber.open(file_obj) as pdf:
            for p in pdf.pages:
                text = p.extract_text() or ""
                pages.append((text, p.page_number))
    except Exception as e:
        # try fallback if pdfplumber fails
        raise RuntimeError(f"Failed to read PDF {getattr(file_obj, 'name', str(file_obj))}: {e}")
    return pages

# ---------------------------
# Simple classifier for document type
# ---------------------------
def classify_document(filename: str) -> str:
    name = filename.lower()
    if any(x in name for x in ["act", "bare", "act,", "act-"]):
        return "acts"
    if any(x in name for x in ["v.", "vs", "versus", "judgment", "judgement", "case", "scc", "supreme"]):
        return "cases"
    if any(x in name for x in ["regulation", "rule", "guideline", "notification", "circular"]):
        return "regulations"
    # heuristic for consumer protection etc.
    if "consumer" in name:
        return "regulations"
    return "others"

# ---------------------------
# Chunking utility
# ---------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = ""
    for p in paragraphs:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = cur + ("\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            if len(p) <= max_chars:
                cur = p
            else:
                # split long paragraph
                start = 0
                while start < len(p):
                    end = start + max_chars
                    chunks.append(p[start:end])
                    start = end - overlap
                cur = ""
    if cur:
        chunks.append(cur)
    # Optionally add tiny overlap slices (already roughly handled)
    return chunks

# ---------------------------
# Embeddings (Gemini) with safe fallbacks
# ---------------------------
def get_embedding(text: str) -> np.ndarray:
    """
    Returns numpy float32 vector.
    Uses google.generativeai's embedding API with a couple fallback checks.
    """
    if not text:
        return np.zeros(1, dtype=np.float32)

    # Try common APIs across versions
    # 1) genai.embeddings.generate (newer)
    try:
        r = genai.embeddings.create(model=EMBED_MODEL_NAME, input=[text])
        # r["data"][0]["embedding"] shape
        emb = r["data"][0]["embedding"]
        return np.array(emb, dtype=np.float32)
    except Exception:
        pass

    # 2) genai.embed_text or genai.embed_content variations
    try:
        # newer wrappers sometimes have function names embed_text / embed_content
        if hasattr(genai, "embed_text"):
            out = genai.embed_text(text, model=EMBED_MODEL_NAME)
            emb = out["embedding"] if isinstance(out, dict) and "embedding" in out else out
            return np.array(emb, dtype=np.float32)
        if hasattr(genai, "embed_content"):
            out = genai.embed_content(model=EMBED_MODEL_NAME, content=text)
            emb = out["embedding"] if isinstance(out, dict) and "embedding" in out else out
            return np.array(emb, dtype=np.float32)
    except Exception:
        pass

    # 3) As last resort try model via GenerativeModel object
    try:
        model = genai.GenerativeModel(EMBED_MODEL_NAME)
        out = model.embed_text(text)
        emb = out["embedding"] if isinstance(out, dict) and "embedding" in out else out
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

# ---------------------------
# FAISS index helpers (manual)
# ---------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build and return a normalized IndexFlatIP (cosine sim via inner product after L2 norm).
    embeddings: np.ndarray shape (N, d)
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def search_index(index: faiss.IndexFlatIP, q_emb: np.ndarray, top_k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    return D, I

# ---------------------------
# Multi-agent DB builder
# ---------------------------
class VectorDB:
    """Holds index, embeddings matrix and metadata list for mapping idx -> metadata+text"""
    def __init__(self, embeddings: np.ndarray, metadatas: List[Dict[str,Any]], texts: List[str]):
        self.embeddings = embeddings.astype(np.float32)
        self.index = build_faiss_index(self.embeddings.copy())
        self.metadatas = metadatas
        self.texts = texts
        self.dim = self.embeddings.shape[1]

    def query(self, q_emb: np.ndarray, top_k: int = 4):
        if self.index.ntotal == 0:
            return []
        D, I = search_index(self.index, q_emb.reshape(1, -1), top_k)
        D = D[0]
        I = I[0]
        results = []
        for score, idx in zip(D, I):
            if idx < 0:
                continue
            meta = self.metadatas[int(idx)]
            text = self.texts[int(idx)]
            results.append({"score": float(score), "metadata": meta, "text": text, "idx": int(idx)})
        return results

# ---------------------------
# Build three DB objects from uploaded PDFs
# ---------------------------
def create_vector_dbs(pdf_files) -> Dict[str, Any]:
    """
    pdf_files: list of uploaded file-like objects (Streamlit's uploaded files)
    Returns: dict with keys 'acts','cases','regulations' mapping to VectorDB or None
    """
    buckets = {"acts": [], "cases": [], "regulations": []}   # lists of (text, metadata)

    splitter_chunk_size = 1200
    splitter_overlap = 150

    for f in pdf_files:
        f_name = getattr(f, "name", str(f))
        dtype = classify_document(f_name)
        pages = extract_pages(f)
        # make page-level chunks
        for page_text, page_no in pages:
            # chunk page_text
            page_chunks = chunk_text(page_text, max_chars=splitter_chunk_size, overlap=splitter_overlap)
            for ci, chunk in enumerate(page_chunks):
                md = {"source": f_name, "page": page_no, "chunk_id": f"{f_name}__p{page_no}__c{ci}", "type": dtype}
                if dtype in buckets:
                    buckets[dtype].append((chunk, md))
                else:
                    # treat "others" as regulations for safety
                    buckets["regulations"].append((chunk, md))

    # For each bucket compute embeddings and build VectorDB
    vector_dbs = {}
    for k in ["acts", "cases", "regulations"]:
        items = buckets[k]
        if not items:
            vector_dbs[k] = None
            continue
        texts = [it[0] for it in items]
        metadatas = [it[1] for it in items]
        # compute embeddings in batches (to avoid huge single call)
        EMB_BATCH = 16
        embs = []
        for i in range(0, len(texts), EMB_BATCH):
            batch_texts = texts[i:i+EMB_BATCH]
            for t in batch_texts:
                v = get_embedding(t)
                embs.append(v)
        embeddings = np.vstack(embs).astype(np.float32)
        vector_dbs[k] = VectorDB(embeddings, metadatas, texts)

    return vector_dbs

# ---------------------------
# Multi-agent query and final synthesis
# ---------------------------
def query_agent(vector_dbs: Dict[str,Any], query: str, top_k:int=3) -> Dict[str, List[Dict]]:
    # compute query embedding
    q_emb = get_embedding(query)
    results = {}
    for k in ["acts", "cases", "regulations"]:
        db = vector_dbs.get(k)
        if db is None:
            results[k] = []
        else:
            results[k] = db.query(q_emb, top_k)
    return results

def synthesize_answer(query: str, retrieved: Dict[str, List[Dict]], max_length: int = 512) -> str:
    """
    Build a prompt from retrieved evidence and ask Gemini to produce final answer with citations.
    """
    # flatten evidence with labels and citations
    evidence_lines = []
    counter = 1
    mapping = {}  # map number -> metadata
    for cat in ["acts", "cases", "regulations"]:
        for hit in retrieved.get(cat, []):
            num = counter
            md = hit["metadata"]
            text = hit["text"]
            citation = f"[{num}] {md['source']} | page {md['page']} | {md['chunk_id']}"
            evidence_lines.append(f"{citation}\n{text}")
            mapping[num] = md
            counter += 1

    if not evidence_lines:
        return "Not available in the uploaded documents."

    context_block = "\n\n---\n\n".join(evidence_lines[:20])  # limit to first 20 hits to keep prompt small

    prompt = f"""
You are a legal assistant. Answer the user's question using ONLY the evidence below. Cite each factual statement by referring to the evidence number in square brackets (e.g., [1], [2]). If the evidence doesn't contain the answer, say "Not available in the uploaded documents."

Question:
{query}

Evidence:
{context_block}

Answer concisely and include citations.
"""

    # Try multiple SDK call styles for broad compatibility
    try:
        model = genai.GenerativeModel(CHAT_MODEL_NAME)
        # newer API: generate_text / generate_content
        # Some SDK versions return object with .text or choices
        resp = model.generate_text(prompt, max_output_tokens=max_length, temperature=0.0)
        # resp may be string or object
        if isinstance(resp, str):
            return resp.strip()
        # if resp is an object try common fields
        if hasattr(resp, "text"):
            return resp.text.strip()
        if isinstance(resp, dict) and "candidates" in resp:
            return (resp["candidates"][0].get("content") or resp["candidates"][0].get("text") or str(resp)).strip()
        # fallback
        return str(resp).strip()
    except Exception:
        # Fallback to older top-level generate function
        try:
            resp = genai.generate_text(model=CHAT_MODEL_NAME, prompt=prompt, max_output_tokens=max_length, temperature=0.0)
            if isinstance(resp, str):
                return resp.strip()
            if isinstance(resp, dict) and "candidates" in resp:
                return (resp["candidates"][0].get("content") or resp["candidates"][0].get("text") or str(resp)).strip()
            return str(resp).strip()
        except Exception as e:
            # as ultimate fallback just concatenate evidence and label them
            simple = "EVIDENCE:\n\n" + "\n\n".join(evidence_lines[:10])
            return f"(Synthesis failed: {e})\n\n{simple}"

# Convenience wrapper
def answer_query(vector_dbs: Dict[str,Any], user_query: str) -> str:
    retrieved = query_agent(vector_dbs, user_query, top_k=3)
    answer = synthesize_answer(user_query, retrieved)
    return answer


