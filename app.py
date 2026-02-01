import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGConfig, read_pdfs_to_docs, chunk_docs, get_vectordb, index_documents, as_retriever
from rag_graph import build_rag_graph

load_dotenv()

st.set_page_config(page_title="RAG Révision", page_icon="📚", layout="wide")
st.title("📚 Assistant IA de révision (RAG)")

# ---- Config env
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")

cfg = RAGConfig(chroma_dir=CHROMA_DIR)

# ---- Session state
if "vectordb" not in st.session_state:
    st.session_state.vectordb = get_vectordb(cfg)

if "graph" not in st.session_state:
    st.session_state.graph = build_rag_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.header("⚙️ Paramètres")
model_name = st.sidebar.text_input("Modèle Groq", value=DEFAULT_MODEL)
k = st.sidebar.slider("k (passages récupérés)", 2, 8, 4)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Réinitialiser la base (Chroma)"):
    # Supprime la base persistée (utile pendant dev)
    try:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        st.session_state.vectordb = get_vectordb(cfg)
        st.success("Base Chroma réinitialisée ✅")
    except Exception as e:
        st.error(f"Erreur reset: {e}")

st.markdown("### 1) Upload des PDFs")
uploaded = st.file_uploader("Ajoute tes PDFs (ex: ceux que tu as générés)", type=["pdf"], accept_multiple_files=True)

colA, colB = st.columns([1, 2])
with colA:
    if st.button("🔎 Indexer", disabled=not uploaded):
        with st.spinner("Lecture → découpage → embeddings → indexation…"):
            docs = read_pdfs_to_docs(uploaded)
            if not docs:
                st.error("Aucun texte lisible trouvé dans les PDFs (si PDF scanné, il faut OCR).")
            else:
                chunks = chunk_docs(docs, cfg)
                n = index_documents(st.session_state.vectordb, chunks)
                st.success(f"Indexation terminée ✅  ({n} chunks ajoutés)")

with colB:
    st.info(
        "Conseil démo : uploade les 4 PDFs (IA, ML, Algo, OS), indexe, puis pose des questions. "
        "Le bot répondra avec une section 'Sources'."
    )

st.markdown("### 2) Poser des questions")
question = st.text_input("Ta question", placeholder="Ex: Explique la différence entre processus et thread.")

if st.button("➡️ Répondre", disabled=not question.strip()):
    # retriever
    retriever = as_retriever(st.session_state.vectordb, k=k)

    with st.spinner("Retrieval + génération…"):
        result = st.session_state.graph.invoke(
            {
                "question": question.strip(),
                "retriever": retriever,
                "k": k,
                "model_name": model_name,
                "messages": st.session_state.messages,
            }
        )
        st.session_state.messages = result.get("messages", st.session_state.messages)
        answer = result.get("answer", "")

    st.subheader("✅ Réponse")
    st.write(answer)

st.markdown("### 🧾 Historique")
for m in st.session_state.messages[-10:]:
    role = "👤" if m.type == "human" else "🤖" if m.type == "ai" else "ℹ️"
    st.markdown(f"**{role} {m.type.upper()}** : {m.content}")
