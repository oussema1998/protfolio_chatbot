import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import FastEmbedEmbeddings

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

DB_DIR = "chroma_db"


def load_or_create_db():
    embeddings = FastEmbedEmbeddings()

    if os.path.exists(DB_DIR):
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
    else:
        loader = PyPDFLoader("cv.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=DB_DIR
        )
        db.persist()

    return db


def create_qa_chain():
    db = load_or_create_db()
    retriever = db.as_retriever()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa