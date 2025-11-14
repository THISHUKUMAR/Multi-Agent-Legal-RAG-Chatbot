import pdfplumber
import faiss
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter  # singular is correct
from langchain.schema import Document
from langchain_community.vectorstores import FAISS


os.environ["GOOGLE_API_KEY"] = "AIzaSyDYAk7r_yA3X4Ir0JCp-pH0rGGtFUB5oRg"
GEMINI_API_KEY = os.environ["GOOGLE_API_KEY"]

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1
)


# ---------------------------
#   PDF TEXT EXTRACTOR WITH PAGE NUMBERS
# ---------------------------
def extract_pages(file):
    pages = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append((text, page.page_number))
    return pages


#   SIMPLE DOCUMENT CLASSIFIER
def classify_document(filename):
    name = filename.lower()

    if any(x in name for x in ["act", "bare", "constitution", "code"]):
        return "acts"
    if any(x in name for x in ["vs", "v.", "judgment", "case", "court"]):
        return "cases"
    if any(x in name for x in ["regulation", "rule", "guideline"]):
        return "regulations"

    return "others"


#   BUILD 3 VECTOR DATABASES
def create_vector_dbs(pdf_files):
    db_acts, db_cases, db_regulations = [], [], []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )

    for file in pdf_files:
        file_type = classify_document(file.name)
        pages = extract_pages(file)

        docs = []
        for text, page_no in pages:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": file.name, "page": page_no, "type": file_type}
                )
            )

        chunks = splitter.split_documents(docs)

        if file_type == "acts":
            db_acts.extend(chunks)
        elif file_type == "cases":
            db_cases.extend(chunks)
        elif file_type == "regulations":
            db_regulations.extend(chunks)

    # Build FAISS vector DBs
    return {
        "acts": FAISS.from_documents(db_acts, embeddings_model) if db_acts else None,
        "cases": FAISS.from_documents(db_cases, embeddings_model) if db_cases else None,
        "regulations": FAISS.from_documents(db_regulations, embeddings_model) if db_regulations else None
    }



#   MULTI-AGENT QUERY HANDLER
def query_agent(db, query):
    if db is None:
        return []

    docs = db.similarity_search(query, k=3)

    results = []
    for d in docs:
        citation = f"Source: {d.metadata['source']} | Page: {d.metadata['page']}"
        results.append(f"{citation}\n{d.page_content}")
    return results


#   FINAL ANSWER AGENT
def answer_query(vector_dbs, query):

    act_hits = query_agent(vector_dbs["acts"], query)
    case_hits = query_agent(vector_dbs["cases"], query)
    reg_hits = query_agent(vector_dbs["regulations"], query)

    if not (act_hits or case_hits or reg_hits):
        return "Not available in the uploaded documents."

    context = "\n\n".join(act_hits + case_hits + reg_hits)

    prompt = f"""
You are a legal reasoning agent. Combine Acts + Case laws + Regulations to answer.

Context:
{context}

User Question: {query}

Rules:
- Answer using ONLY the given context.
- Provide citations exactly as given above.
- If something is not found in context, DO NOT guess.
"""

    response = llm.invoke(prompt)
    return response.content

