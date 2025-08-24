# Chatbot RAG para consultar el CV del alumno (Streamlit)

Sistema de **Retrieval-Augmented Generation (RAG)** que permite consultar un CV y obtener respuestas con citas de las secciones relevantes.
Listo para ejecutar localmente con **Streamlit + FAISS + Sentence-Transformers** y generación vía **Ollama** (local) u **OpenAI** (SaaS).

## 🚀 Demo rápida (5 pasos)

```bash
# 1) Clonar o copiar esta carpeta
cd rag_cv_chatbot

# 2) Crear entorno e instalar deps
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
pip install -r requirements.txt

# 3) (Opcional) Configurar modelo de generación
#    Opción A - Local con Ollama (recomendado, gratis):
#      - Instalar https://ollama.com/download
#      - Descargar modelo:  ollama pull llama3.1:8b
#      - Exportar variable:  set OLLAMA_MODEL=llama3.1:8b   (Windows)
#                            export OLLAMA_MODEL=llama3.1:8b (Mac/Linux)
#
#    Opción B - OpenAI:
#      - Copiar .env.example a .env y setear OPENAI_API_KEY=<tu_api_key>

# 4) Colocar tu CV en /data como PDF/TXT/MD (ej: cv_alumno.pdf).
#    También podés subirlo desde la UI.
python ingest.py  # crea/actualiza el índice

# 5) Lanzar la app
streamlit run app.py
```

Abre el navegador en la URL que te imprime Streamlit (usualmente `http://localhost:8501`).

---

## 📁 Estructura

```
rag_cv_chatbot/
├─ app.py                # UI Streamlit (chat + upload + reindex)
├─ ingest.py             # Ingesta de documentos de /data -> índice FAISS
├─ rag.py                # Núcleo: embedder, retriever y generación
├─ utils.py              # Carga y troceo de documentos
├─ requirements.txt
├─ .env.example
├─ data/
│  └─ CV_ejemplo.md
├─ storage/              # Se crea al ejecutar ingest.py (índice y metadatos)
└─ docs/
   └─ demo_script.md     # Guion sugerido para tu video de presentación
```

## ✨ Características

- **Ingesta simple**: arrastrá tu CV (PDF/TXT/MD) o reemplazá el ejemplo y ejecutá `python ingest.py`.
- **Embeddings multilingües**: `paraphrase-multilingual-MiniLM-L12-v2` (soporta español).
- **FAISS** como vector store en disco.
- **RAG** con *top-k retrieval* y citas (fragmentos + archivo origen).
- **Modelo de texto** intercambiable: **Ollama local** o **OpenAI** (selección automática por variables de entorno).
- **Streamlit** con historial de chat y subida de archivos.

## 🧠 ¿Cómo funciona?

1. **Troceo del CV** en fragmentos (~500 tokens con solapamiento).
2. **Embeddings** de cada chunk y construcción del índice **FAISS**.
3. En consulta, se recuperan los **k** fragmentos más similares.
4. Se genera una respuesta con un **prompt con contexto** que incluye los fragmentos recuperados.
5. Se muestran **citas** (archivo y preview del texto) para trazabilidad.

## ⚙️ Configuración

- Variables de entorno (en `.env` si usás OpenAI):
  - `OPENAI_API_KEY`: clave para usar modelos de OpenAI.
  - `OPENAI_MODEL` (opcional): por defecto `gpt-4o-mini`.
  - `OLLAMA_MODEL` (opcional): p.ej. `llama3.1:8b` si usás Ollama local.

- Parámetros principales (editables en `rag.py`):
  - `EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
  - `CHUNK_SIZE = 1000` (caracteres aproximados)
  - `CHUNK_OVERLAP = 150`
  - `TOP_K = 4`

## 🧪 Pruebas rápidas de preguntas

- *"¿Cuál es la formación académica del alumno?"*
- *"Enumera experiencias relevantes en QA/IA con fechas."*
- *"¿Qué habilidades técnicas tiene y en qué proyectos las aplicó?"*
- *"¿Datos de contacto?"*

## 🎥 Entregable del video (OBS)

1. Muestra el repo y `README.md`.
2. Ejecutá `python ingest.py` con tu **CV real** en `/data` o subilo desde la UI.
3. Abrí `streamlit run app.py`, hacé 2–3 consultas y mostrá citas.
4. Finalizá explicando brevemente la arquitectura RAG.

## 🧾 Licencia

MIT. Podés reutilizar y modificar libremente citando este repo en tu TP.


---

## 🧑‍💼 Nuevo: Modo **Multi‑Agentes por Persona**

Este repo ahora soporta **1 agente por persona**. Cada agente tiene su **propio índice RAG** (FAISS) limitado a los documentos de su carpeta en `data/<Persona>/`.  
**Reglas clave:**

- Si la consulta **no nombra a nadie**, se usa el **agente del Alumno** (configurable en `config/people.json` → `default_student`).
- Si la consulta **nombra a más de una persona**, se **ejecutan múltiples agentes** y se compone una respuesta con **secciones por persona** y **citas**.
- Estructura esperada de datos:
  ```text
  data/
    Alumno/      # CVs del alumno (por defecto)
    Persona2/
    Persona3/
  ```

### ▶️ Pasos rápidos

```bash
# 0) (opcional) Crear venv + instalar deps
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Editar personas y alias
cp -n config/people.json config/people.json  # ya existe con ejemplo
# Abrí config/people.json y ajustá nombres/alias y default_student

# 2) Colocar CVs (PDF/TXT/MD) en /data/<Persona>/
mkdir -p data/Alumno
cp data/CV_ejemplo.md data/Alumno/  # ejemplo

# 3) Construir índices por persona
python ingest_agents.py

# 4) Ejecutar la app multi‑agentes (Streamlit)
streamlit run app_agents.py
```

> **Nota de evaluación:** El video debe mostrar: consulta por defecto (sin nombre → Alumno), consulta a otra persona nombrada, y consulta a **2 personas a la vez** con respuestas en secciones y citas por persona.

### 🎥 Guion sugerido para el video (≤ 2 min)

1. **Intro (10s):** objetivo del TP y que hay *1 agente por persona*.
2. **Arquitectura (20s):** `data/<Persona> → ingest_agents.py → storage/<Persona> (FAISS) → app_agents.py`.
3. **Demo (60s):**
   - Pregunta sin nombre (“¿Cuál es tu formación?”) → responde Alumno.
   - Pregunta “¿Qué experiencia tienen *Alumno* y *Persona2* en Python?” → muestra secciones por persona + citas.
4. **Cierre (20s):** cómo agregar una tercera persona y repetir los pasos.

