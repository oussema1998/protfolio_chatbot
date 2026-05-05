import os
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

DB_DIR = "chroma_db"
PDF_PATH = "cv.pdf"


# =========================
# 🔥 HF API EMBEDDINGS
# =========================
class HFEmbeddings:
    def __init__(self):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN")

        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.api_urls = [
            f"https://router.huggingface.co/hf-inference/models/{self.model_id}/pipeline/feature-extraction",
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}",
        ]

        self.headers = {"Authorization": f"Bearer {token}"}

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        for url in self.api_urls:
            response = requests.post(
                url,
                headers=self.headers,
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and result and isinstance(result[0], list):
                    return self._mean_pool(result)
                return result

        raise RuntimeError(f"HF API failed: {response.text[:200]}")

    def _mean_pool(self, matrix):
        length = len(matrix)
        pooled = [0.0] * len(matrix[0])

        for row in matrix:
            for i, v in enumerate(row):
                pooled[i] += v

        return [v / length for v in pooled]


def get_embeddings():
    return HFEmbeddings()


# =========================
# 📄 BUILD DB
# =========================
def build_db():
    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("✂️ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    print("🧠 Embeddings...")
    embeddings = get_embeddings()

    print("💾 Creating DB...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )

    print("✅ DB created!")


# =========================
# 📦 LOAD DB (SAFE)
# =========================
def load_db():
    embeddings = get_embeddings()

    if not os.path.exists(DB_DIR):
        print("⚠️ DB not found → building it...")
        build_db()

    print("📂 Loading DB...")
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )


# =========================
# 🤖 QA CHAIN
# =========================
def create_qa_chain():
    db = load_db()

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )


# =========================
# 🚀 LOCAL RUN
# =========================
if __name__ == "__main__":
    build_db()