import streamlit as st
from backend import create_vector_dbs, answer_query

st.set_page_config(page_title="Legal RAG Chatbot", layout="wide")
st.title("⚖ Multi-Agent Legal RAG Chatbot")

# Store Vector DB and Chat History
if "vector_dbs" not in st.session_state:
    st.session_state.vector_dbs = None

if "chat" not in st.session_state:
    st.session_state.chat = []


# --------------------------------------------------------
# Upload PDFs
# --------------------------------------------------------
uploaded = st.file_uploader(
    "Upload Legal PDFs (Acts, Cases, Regulations)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded:
    st.session_state.vector_dbs = create_vector_dbs(uploaded)
    st.success("Vector databases created successfully!")


# --------------------------------------------------------
# Show Chat
# --------------------------------------------------------
st.subheader("Chat")

for role, msg in st.session_state.chat:
    st.chat_message(role).markdown(msg)


# --------------------------------------------------------
# User Query Input
# --------------------------------------------------------
query = st.chat_input("Ask a legal question...")

if query:
    st.session_state.chat.append(("user", query))
    st.chat_message("user").markdown(query)

    if st.session_state.vector_dbs is None:
        answer = "Please upload PDFs first."
    else:
        answer = answer_query(
            st.session_state.vector_dbs,
            query,
            st.session_state.chat
        )

    st.session_state.chat.append(("assistant", answer))
    st.chat_message("assistant").markdown(answer)
