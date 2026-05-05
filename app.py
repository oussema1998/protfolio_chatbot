import streamlit as st
from rag import create_qa_chain, load_db

st.set_page_config(
    page_title="Chatbot CV RAG",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Chatbot CV (RAG)")
st.markdown("Pose des questions sur le CV")

# =========================
# 🔥 INIT SAFE
# =========================
@st.cache_resource
def load_chain():
    st.info("⚙️ Initialisation du système RAG...")

    # force DB creation if needed
    load_db()

    return create_qa_chain()

try:
    qa = load_chain()
    st.success("✅ Système prêt !")
except Exception as e:
    st.error("❌ Erreur au chargement")
    st.error(str(e))
    st.stop()


# =========================
# 💬 CHAT
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("💬 Ta question")

if question:
    with st.spinner("🤔 Analyse du CV..."):
        try:
            response = qa.run(question)
            st.session_state.history.append((question, response))
        except Exception as e:
            st.error(str(e))


# =========================
# 📜 HISTORY
# =========================
if st.session_state.history:
    st.divider()
    st.subheader("🧾 Historique")

    for q, r in reversed(st.session_state.history):
        st.markdown(f"**🧑 {q}**")
        st.markdown(f"**🤖 {r}**")
        st.divider()