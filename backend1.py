import pdfplumber
import faiss
import numpy as np
import os
import google.generativeai as genai

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------
#   GEMINI CONFIG
# ---------------------------

os.environ["GOOGLE_API_KEY"] = "AIzaSyCUIycD0goSuABem0Aungs95Lt_rkM6fa8"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

embed_model = genai.GenerativeModel("text-embedding-004")
chat_model = genai.GenerativeModel("gemini-2.0-flash")

# ---------------------------
#   PDF TEXT EXTRACTOR
# ---------------------------
def extract_pages(file):
    pages = []
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            text = p.extract_text() or ""
            pages.append((text, p.page_number))
    return pages


# ---------------------------
#   CLASSIFIER
# ---------------------------
def classify_document(filename):
    name = filename.lower()

    if any(x in name for x in ["act", "constitution", "code"]):
        return "acts"
    if any(x in name for x in ["vs", "v.", "case", "judgment"]):
        return "cases"
    if any(x in name for x in ["regulation", "rule", "guideline"]):
        return "regulations"

    return "others"


# ---------------------------
#   BUILD VECTOR DBs
# ---------------------------
def embed_text(t):
    out = embed_model.embed_content(t)
    return np.array(out["embedding"], dtype=np.float32)

def create_vector_dbs(pdf_files):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

    acts, cases, regs = [], [], []

    for f in pdf_files:
        dtype = classify_document(f.name)
        pages = extract_pages(f)

        docs = [
            Document(
                page_content=text,
                metadata={"source": f.name, "page": page_no, "type": dtype},
            )
            for text, page_no in pages
        ]

        chunks = splitter.split_documents(docs)

        if dtype == "acts":
            acts.extend(chunks)
        elif dtype == "cases":
            cases.extend(chunks)
        elif dtype == "regulations":
            regs.extend(chunks)

    def build_db(docs):
        if not docs:
            return None
        texts = [d.page_content for d in docs]
        embeddings = [embed_text(t) for t in texts]
        return FAISS.from_embeddings(embeddings, docs)

    return {
        "acts": build_db(acts),
        "cases": build_db(cases),
        "regulations": build_db(regs)
    }


# ---------------------------
#   SEARCH + FINAL ANSWER
# ---------------------------
def query_agent(db, q):
    if db is None:
        return []

    q_emb = embed_text(q)
    docs = db.similarity_search_by_vector(q_emb, k=3)

    results = []
    for d in docs:
        citation = f"Source: {d.metadata['source']} | Page: {d.metadata['page']}"
        results.append(f"{citation}\n{d.page_content}")

    return results


def answer_query(vector_dbs, query):
    act_hits = query_agent(vector_dbs["acts"], query)
    case_hits = query_agent(vector_dbs["cases"], query)
    reg_hits = query_agent(vector_dbs["regulations"], query)

    context = "\n\n".join(act_hits + case_hits + reg_hits)

    if not context:
        return "Not found in uploaded documents."

    prompt = f"""
Use ONLY the context.

Context:
{context}

Question: {query}

Give a legal answer with citations.
"""

    response = chat_model.generate_content(prompt)
    return response.text
