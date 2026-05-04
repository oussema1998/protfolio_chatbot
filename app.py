import streamlit as st
from rag import create_qa_chain

st.title("🤖 Chatbot CV RAG")

@st.cache_resource
def load_chain():
    return create_qa_chain()

qa = load_chain()

question = st.text_input("Ta question")

if question:
    response = qa.run(question)
    st.write(response)