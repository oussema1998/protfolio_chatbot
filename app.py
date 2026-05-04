import streamlit as st
from rag import create_qa_chain

st.title("🤖 Chatbot CV RAG")

@st.cache_resource
def load_chain():
    return create_qa_chain()

try:
    qa = load_chain()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

question = st.text_input("Pose ta question")

if question:
    with st.spinner("Réflexion..."):
        response = qa.run(question)
        st.write(response)