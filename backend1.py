import pdfplumber
import faiss
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


os.environ["GOOGLE_API_KEY"] = "AIzaSyCUIycD0goSuABem0Aungs95Lt_rkM6fa8"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Embedding model
EMBEDDING_MODEL = "models/text-embedding-004"

def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
    )
    return result["embedding"]
    

def extract_pages(file):
    pages = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append((text, page.page_number))
    return pages


def classify_document(filename):
    name = filename.lower()
    if any(x in name for x in ["act", "bare", "constitution", "code"]):
        return "acts"
    if any(x in name for x in ["vs", "v.", "judgment", "case", "court"]):
        return "cases"
    if any(x in name for x in ["regulation", "rule", "guideline"]):
        return "regulations"
    return "others"


def create_vector_dbs(pdf_files):
    db = {"acts": [], "cases": [], "regulations": []}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )

    for file in pdf_files:
        file_type = classify_document(file.name)
        pages = extract_pages(file)

        docs = []
        for text, p in pages:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file.name, "page": p, "type": file_type},
                )
            )

        chunks = splitter.split_documents(docs)
        db[file_type].extend(chunks)

    vector_dbs = {}
    for key in db:
        if db[key]:
            vector_dbs[key] = FAISS.from_embeddings(
                [(d.page_content, get_embedding(d.page_content), d.metadata) for d in db[key]],
                embedding_size=768
            )
        else:
            vector_dbs[key] = None

    return vector_dbs


def query_agent(db, query):
    if db is None:
        return []

    emb = get_embedding(query)
    docs = db.similarity_search_by_vector(emb, k=3)

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
        return "Not available in the uploaded documents."

    prompt = f"""
You are a legal agent. Answer using ONLY this context:

{context}

User Question: {query}

Rules:
- Do not hallucinate.
- Use exact citations from text.
"""

    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(prompt)
    return response.text
