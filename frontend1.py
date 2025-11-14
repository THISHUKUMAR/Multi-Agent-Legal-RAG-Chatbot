# frontend.py
# Streamlit UI for the Multi-Agent Legal RAG Chatbot (uses backend.py)

import streamlit as st
from backend import create_vector_dbs, answer_query

st.set_page_config(page_title="Multi-Agent Legal RAG", layout="wide")
st.title("⚖ Multi-Agent Legal RAG Chatbot")

# Initialize session state
if "vector_dbs" not in st.session_state:
    st.session_state.vector_dbs = None
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role":"user"/"assistant","content":...}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Left: upload and process
with st.sidebar:
    st.header("Upload & Build KB")
    uploaded_files = st.file_uploader(
        "Upload Bare Acts, Case Laws, Regulations (PDF). Select multiple.",
        type=["pdf"], accept_multiple_files=True
    )
    if st.button("Process Documents"):
        if not uploaded_files:
            st.sidebar.error("Upload at least one PDF")
        else:
            with st.spinner("Processing PDFs and building vector DBs (this can take a minute)..."):
                try:
                    st.session_state.vector_dbs = create_vector_dbs(uploaded_files)
                    st.session_state.processed_files = [getattr(f,"name",str(f)) for f in uploaded_files]
                    st.success(f"Knowledge base created from {len(uploaded_files)} files.")
                except Exception as e:
                    st.error(f"Processing failed: {e}")

    if st.session_state.processed_files:
        st.markdown("**Processed files:**")
        for fn in st.session_state.processed_files:
            st.markdown(f"- {fn}")

st.markdown("---")
st.subheader("Chat with your legal documents")
st.caption("Ask legal questions; the system uses Acts, Cases & Regulations separately and cites sources.")

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

# Chat input
user_input = st.chat_input("Ask a legal question…")

if user_input:
    # append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    # show it immediately
    st.experimental_rerun()  # ensures message shows before long processing (optional); Streamlit may re-run

# The rerun will cause frontend to re-execute; we now detect last user message without assistant reply
if st.session_state.messages:
    # find last pair: if last role is user, we need to process
    last = st.session_state.messages[-1]
    if last["role"] == "user" and (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "assistant"):
        user_q = last["content"]
        # if KB not ready
        if st.session_state.vector_dbs is None:
            reply = "Please upload and process documents first (use the sidebar)."
        else:
            with st.spinner("Thinking…"):
                try:
                    reply = answer_query(st.session_state.vector_dbs, user_q)
                except Exception as e:
                    reply = f"Error generating answer: {e}"
        # append assistant reply
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.experimental_rerun()
