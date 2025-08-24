import os, json, pathlib
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# Parámetros principales
# -----------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
INDEX_DIR = pathlib.Path("storage")
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "meta.json"

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts: List[str]):
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def ensure_index_exists():
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError("No existe el índice. Ejecuta primero: python ingest.py")

def retrieve(query: str) -> List[Dict[str, Any]]:
    ensure_index_exists()
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    dim = meta["dim"]
    index = faiss.read_index(str(INDEX_FILE))
    # Embedding de la query
    qv = embed_texts([query])
    D, I = index.search(qv, TOP_K)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1: 
            continue
        item = meta["chunks"][idx]
        item["score"] = float(D[0][rank])
        results.append(item)
    return results

# -----------------------
# Generación
# -----------------------
def generate_answer(prompt: str) -> str:
    # Opción A: Ollama local
    ollama_model = os.getenv("OLLAMA_MODEL")
    if ollama_model:
        try:
            import ollama
            # Mantenerlo simple; se puede ajustar temperature, etc.
            resp = ollama.chat(model=ollama_model, messages=[{"role":"user","content":prompt}])
            return resp["message"]["content"]
        except Exception as e:
            return f"[ERROR OLLAMA] {e}"

    # Opción B: OpenAI
    from openai import OpenAI
    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"Sos un asistente conciso y preciso. Responde en español y cita las fuentes provistas cuando corresponda."},
                {"role":"user","content": prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[ERROR OPENAI] {e}"

def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        [f"[{i+1}] (archivo: {c['source']})\n{c['text']}" for i,c in enumerate(contexts)]
    )
    return f"""Usando exclusivamente la siguiente información de contexto (fragmentos recuperados del CV), respondé en español de manera directa y breve. 
Si no está en los textos, decí honestamente que no aparece en el CV.

Pregunta: {query}

Contexto:
{context_block}

Instrucciones:
- Integrá y sintetizá la información relevante.
- Al final, añadí una sección "Fuentes" con los índices de fragmentos usados (ej: [1], [3]) y el nombre de archivo.
"""

def rag_answer(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    contexts = retrieve(query)
    prompt = build_prompt(query, contexts)
    answer = generate_answer(prompt)
    return answer, contexts
