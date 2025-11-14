import streamlit as st
from backend1 import create_vector_dbs, answer_query

st.set_page_config(page_title="Multi-Agent Legal RAG System", layout="wide")
st.title("⚖ Multi-Agent Legal RAG Chatbot")


# Initialize session state

if "vector_dbs" not in st.session_state:
    st.session_state.vector_dbs = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # store chat history


# File Upload Section
uploaded = st.file_uploader(
    "Upload Bare Acts, Case Laws, Regulations",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process Documents"):
    if not uploaded:
        st.error("Upload at least one PDF.")
    else:
        with st.spinner("Building multi-agent knowledge base…"):
            st.session_state.vector_dbs = create_vector_dbs(uploaded)
        st.success("Knowledge base created successfully!")


# Display Previous Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# Chat Input
user_query = st.chat_input("Ask a legal question...")

if user_query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get assistant response
    if st.session_state.vector_dbs is None:
        reply = "Please upload and process documents first."
    else:
        with st.spinner("Thinking…"):
            reply = answer_query(st.session_state.vector_dbs, user_query)

    # Add assistant reply to history
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Refresh page to show latest messages
    st.rerun()
