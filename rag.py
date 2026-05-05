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
# 🔥 HF API EMBEDDINGS (NO LOCAL MODEL)
# =========================
class HFEmbeddings:
    def __init__(self):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("Missing HUGGINGFACEHUB_API_TOKEN in environment")
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.api_urls = [
            "https://router.huggingface.co/hf-inference/"
            f"models/{self.model_id}/pipeline/feature-extraction",
            "https://api-inference.huggingface.co/"
            f"pipeline/feature-extraction/{self.model_id}",
            "https://router.huggingface.co/hf-inference/"
            f"models/{self.model_id}",
            "https://api-inference.huggingface.co/"
            f"models/{self.model_id}",
        ]
        self.headers = {
            "Authorization": f"Bearer {token}"
        }

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        last_error = None
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        for url in self.api_urls:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 404:
                last_error = response.text[:500]
                continue

            if response.status_code == 400 and "sentences" in response.text:
                # This means the endpoint expects sentence-similarity inputs.
                last_error = response.text[:500]
                continue

            if not response.ok:
                # Surface API errors and non-JSON responses clearly.
                snippet = response.text[:500]
                raise RuntimeError(
                    f"HF API request failed ({response.status_code}): {snippet}"
                )

            try:
                result = response.json()
            except requests.exceptions.JSONDecodeError:
                snippet = response.text[:500]
                raise RuntimeError(
                    f"HF API returned non-JSON response: {snippet}"
                )

            if isinstance(result, dict) and "error" in result:
                raise Exception(f"HF API error: {result['error']}")

            if isinstance(result, list) and result and isinstance(result[0], list):
                # Mean-pool token embeddings into a single vector.
                return self._mean_pool(result)

            return result

        raise RuntimeError(
            "HF API endpoint not found. "
            f"Tried {len(self.api_urls)} endpoints. Last response: {last_error}"
        )

    def _mean_pool(self, matrix):
        length = len(matrix)
        if length == 0:
            return []
        pooled = [0.0] * len(matrix[0])
        for row in matrix:
            for i, value in enumerate(row):
                pooled[i] += value
        return [value / length for value in pooled]


def get_embeddings():
    return HFEmbeddings()


# =========================
# 📄 BUILD VECTOR DB (RUN ONCE LOCALLY)
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

    print("🧠 Calling Hugging Face API embeddings...")
    embeddings = get_embeddings()

    print("💾 Creating Chroma DB...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )

    db.persist()
    print("✅ DB built successfully!")


# =========================
# 📦 LOAD DB
# =========================
def load_db():
    if not os.path.exists(DB_DIR):
        raise RuntimeError("❌ DB not found. Run build_db() first.")

    print("📂 Loading DB...")
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )


# =========================
# 🤖 QA CHAIN (GROQ LLM)
# =========================
def create_qa_chain():
    db = load_db()

    retriever = db.as_retriever(search_kwargs={"k": 3})

    print("🤖 Loading Groq LLM...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )


# =========================
# 🚀 ENTRY POINT (BUILD DB)
# =========================
if __name__ == "__main__":
    build_db()