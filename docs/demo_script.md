# Guion sugerido para el video (OBS)

1. **Introducción (10–15s)**
   - Presentate y menciona que el TP implementa un Chatbot RAG para consultar tu CV.

2. **Breve arquitectura (15–20s)**
   - Explicá: embeddings + FAISS + top-k retrieval + generación con LLM (Ollama u OpenAI).

3. **Ejecución (40–60s)**
   - Mostrar consola: `python ingest.py` y creación del índice.
   - Correr `streamlit run app.py` y abrir la UI.
   - Hacer 2–3 preguntas típicas: formación, experiencia clave, skills.
   - Mostrar el panel de citas/fragmentos.

4. **Cierre (10–15s)**
   - Resumí beneficios (respuestas con respaldo y trazabilidad) y cómo extender a más documentos.
