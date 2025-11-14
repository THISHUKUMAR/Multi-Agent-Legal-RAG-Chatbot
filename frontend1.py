import pdfplumber
import numpy as np
import faiss
import google.generativeai as genai


genai.configure(api_key="YOUR_API_KEY")


# --------------------------------------------------------
# Embedding model
# --------------------------------------------------------
embed_model = genai.GenerativeModel("models/text-embedding-004")


def embed_text(text: str):
    emb = embed_model.embed_content(text)["embedding"]
    return np.array(emb, dtype=np.float32)


# --------------------------------------------------------
# PDF extraction
# --------------------------------------------------------
def extract_pages(file):
    pages = []
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            text = p.extract_text() or ""
            pages.append((text, p.page_number))
    return pages


# --------------------------------------------------------
# Document classifier
# --------------------------------------------------------
def classify_document(filename):
    name = filename.lower()

    if any(x in name for x in ["act", "bare", "constitution", "code"]):
        return "acts"
    if any(x in name for x in ["vs", "v.", "judgment", "case"]):
        return "cases"
    if any(x in name for x in ["regulation", "rule", "guideline"]):
        return "regulations"

    return "others"


# --------------------------------------------------------
# Build FAISS DB
# --------------------------------------------------------
def build_db(docs):
    if not docs:
        return None

    embeddings = [embed_text(d["text"]) for d in docs]

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return {
        "index": index,
        "docs": docs
    }


# --------------------------------------------------------
# Create vector DBs
# --------------------------------------------------------
def create_vector_dbs(pdf_files):
    acts, cases, regs = [], [], []

    for f in pdf_files:
        f_type = classify_document(f.name)
        pages = extract_pages(f)

        for text, pg in pages:
            entry = {"text": text, "page": pg, "file": f.name}

            if f_type == "acts":
                acts.append(entry)
            elif f_type == "cases":
                cases.append(entry)
            elif f_type == "regulations":
                regs.append(entry)

    return {
        "acts": build_db(acts),
        "cases": build_db(cases),
        "regulations": build_db(regs)
    }


# --------------------------------------------------------
# Query vector DB
# --------------------------------------------------------
def search_db(db, query, k=3):
    if db is None:
        return []

    q_emb = embed_text(query).reshape(1, -1)
    D, I = db["index"].search(q_emb, k)

    res = []
    for idx in I[0]:
        if idx < 0:
            continue
        d = db["docs"][idx]
        citation = f"Source: {d['file']} | Page: {d['page']}"
        res.append(f"{citation}\n{d['text']}")
    return res


# --------------------------------------------------------
# Final answer generator
# --------------------------------------------------------
def answer_query(vector_dbs, query, chat_history):
    acts = search_db(vector_dbs["acts"], query)
    cases = search_db(vector_dbs["cases"], query)
    regs = search_db(vector_dbs["regulations"], query)

    context = "\n\n".join(acts + cases + regs)

    if not context:
        return "No matching content found in uploaded PDFs."

    prompt = f"""
You are a legal reasoning assistant.
Use ONLY the following context to answer.

Context:
{context}

User's Question:
{query}

Chat History:
{chat_history}

Rules:
- Cite exactly as given.
- Do NOT guess outside the context.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)

    return resp.text
