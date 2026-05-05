import streamlit as st
from rag import create_qa_chain

# =========================
# ⚙️ CONFIG
# =========================
st.set_page_config(
    page_title="Chatbot CV RAG",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Chatbot CV (RAG)")

st.markdown(
    "Pose des questions sur le CV. Exemple : *quelles sont ses compétences ?*"
)

# =========================
# 📦 LOAD QA CHAIN (CACHED)
# =========================
@st.cache_resource
def load_chain():
    st.info("⚙️ Initialisation du système RAG...")
    return create_qa_chain()

try:
    qa = load_chain()
    st.success("✅ Système prêt !")
except Exception as e:
    st.error("❌ Erreur lors du chargement du modèle")
    st.error(str(e))
    st.stop()

# =========================
# 💬 CHAT UI
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("💬 Ta question sur le CV")

# =========================
# 🤖 RESPONSE
# =========================
if question:
    with st.spinner("🤔 Analyse du CV..."):
        try:
            response = qa.run(question)

            # save history
            st.session_state.history.append((question, response))

        except Exception as e:
            st.error(f"Erreur: {e}")

# =========================
# 📜 HISTORY DISPLAY
# =========================
if st.session_state.history:
    st.divider()
    st.subheader("🧾 Historique")

    for q, r in reversed(st.session_state.history):
        st.markdown(f"**🧑 Question :** {q}")
        st.markdown(f"**🤖 Réponse :** {r}")
        st.divider()