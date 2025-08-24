import os, pathlib, time
import streamlit as st
from dotenv import load_dotenv
from rag import rag_answer, ensure_index_exists
from utils import load_text_from_file
import shutil

load_dotenv()

st.set_page_config(page_title="Chatbot RAG - CV Alumno", page_icon="🧠", layout="wide")
st.title("🧠 Chatbot RAG – Consulta el CV del alumno")

# Sidebar
st.sidebar.header("⚙️ Configuración")
st.sidebar.write("Motor de generación:")
ollama_model = os.getenv("OLLAMA_MODEL")
openai_key = os.getenv("OPENAI_API_KEY")
if ollama_model:
    st.sidebar.success(f"Ollama activo: `{ollama_model}`")
elif openai_key:
    st.sidebar.success("OpenAI activo (API Key detectada)")
else:
    st.sidebar.warning("No hay motor configurado. Seteá OLLAMA_MODEL o OPENAI_API_KEY.")

# Upload de documento
st.sidebar.subheader("📄 Subir/actualizar CV")
uploaded = st.sidebar.file_uploader("Arrastrá tu CV (PDF/TXT/MD)", type=["pdf","txt","md"])
if uploaded is not None:
    data_dir = pathlib.Path("data")
    data_dir.mkdir(exist_ok=True)
    dest = data_dir / uploaded.name
    with open(dest, "wb") as f:
        f.write(uploaded.read())
    st.sidebar.success(f"Guardado: {dest}")
    st.sidebar.info("Ejecutá `python ingest.py` en tu terminal o presioná el botón abajo para reindexar.")

# Botón reindexar
if st.sidebar.button("🔄 Reindexar ahora"):
    with st.spinner("Construyendo índice..."):
        import subprocess, sys
        # Ejecutar ingest.py en subproceso para logs limpios
        result = subprocess.run([sys.executable, "ingest.py"], capture_output=True, text=True)
        st.sidebar.code(result.stdout + ("\n" + result.stderr if result.stderr else ""))
    st.sidebar.success("Reindexación completa.")

# Chat
st.subheader("💬 Consulta")
if "history" not in st.session_state:
    st.session_state["history"] = []

user_query = st.chat_input("Escribe tu pregunta sobre el CV...")
if user_query:
    try:
        ensure_index_exists()
        with st.spinner("Buscando en el CV y redactando respuesta..."):
            answer, ctxs = rag_answer(user_query)
        st.session_state["history"].append({"role":"user", "content": user_query})
        st.session_state["history"].append({"role":"assistant", "content": answer, "ctxs": ctxs})
    except Exception as e:
        st.error(f"No se pudo responder: {e}")

for msg in st.session_state["history"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            ctxs = msg.get("ctxs", [])
            if ctxs:
                with st.expander("🔎 Ver fragmentos citados"):
                    for i, c in enumerate(ctxs, start=1):
                        st.markdown(f"**[{i}] Fuente:** `{c['source']}` — *score:* {c['score']:.3f}")
                        st.code(c["text"][:1200])
