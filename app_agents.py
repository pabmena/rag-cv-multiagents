
import os, pathlib, time
import streamlit as st
from dotenv import load_dotenv
from agents import PersonAgent, list_people, resolve_people_from_query, get_default_student
from utils import load_text_from_file
import shutil, json

load_dotenv()

st.set_page_config(page_title="Chatbot RAG Multi-Agentes (CVs)", page_icon="🧑‍💼", layout="wide")
st.title("🧑‍💼 Chatbot RAG Multi‑Agentes – CVs del equipo")

# Sidebar
st.sidebar.header("⚙️ Configuración")
st.sidebar.write("Este chatbot crea **un agente por persona** (uno por carpeta en /data).")

# People panel
st.sidebar.subheader("👤 Personas detectadas")
people = list_people()
if not people:
    st.sidebar.warning("No hay personas en config/people.json")
else:
    st.sidebar.write(", ".join(people))
st.sidebar.write(f"Por defecto (sin nombre en la query): **{get_default_student()}**")

# Upload panel
st.sidebar.subheader("📥 Cargar documentos")
with st.sidebar.form("upload_form"):
    person_sel = st.selectbox("Persona", options=people or ["Alumno"])
    up = st.file_uploader("Archivo (PDF/TXT/MD)", type=["pdf","txt","md"])
    submitted = st.form_submit_button("Guardar en /data/<Persona>/")
    if submitted:
        if up is None:
            st.sidebar.error("Seleccioná un archivo")
        else:
            out_dir = pathlib.Path("data")/person_sel
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / up.name
            with open(out_path, "wb") as f:
                f.write(up.read())
            st.sidebar.success(f"Guardado en {out_path}. Recordá reconstruir el índice:")
            st.sidebar.code("python ingest_agents.py")

# Session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Chat input
query = st.chat_input("Escribí tu pregunta (podés nombrar 1 o más personas)")
if query:
    st.session_state["history"].append({"role":"user", "content":query})

# Render chat
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            bundle = msg.get("bundle", [])
            if bundle:
                with st.expander("🔎 Ver fragmentos citados por persona"):
                    for person, ctxs in bundle:
                        st.markdown(f"**👤 {person}**")
                        for i, c in enumerate(ctxs, start=1):
                            st.markdown(f"**[{i}]** `{c['source']}` — *score:* {c['score']:.3f}")
                            st.code(c['text'][:1000])

# On new user message → route to agents
if query:
    target_people = resolve_people_from_query(query)
    st.info(f"Agentes seleccionados: {', '.join(target_people)}")
    bundle = []
    parts = []
    for person in target_people:
        try:
            agent = PersonAgent(person)
            ans, ctxs = agent.answer(query)
            bundle.append((person, ctxs))
            parts.append(f"""### {person}
{ans}
""")
        except Exception as e:
            parts.append(f"""### {person}
_No se pudo responder para **{person}**: {e}_""")
    combined = "\n\n".join(parts)
    st.session_state["history"].append({"role":"assistant", "content": combined, "bundle": bundle})
    st.experimental_rerun()
