import os, json, pathlib
from typing import List, Dict, Any
import faiss
from rag import embed_texts, get_embedder, INDEX_DIR, INDEX_FILE, META_FILE
from utils import iter_docs, chunk_text

DATA_DIR = pathlib.Path("data")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_index():
    docs = iter_docs(str(DATA_DIR))
    if not docs:
        print("[ERROR] No hay documentos en /data. Copiá tu CV allí (PDF/TXT/MD).")
        return

    print(f"[INFO] Documentos a indexar: {len(docs)}")
    chunks = []
    mapping = []
    for path, content in docs:
        parts = chunk_text(content, chunk_size=1000, overlap=150)
        for i, ch in enumerate(parts):
            chunks.append(ch)
            mapping.append({"source": os.path.basename(path), "chunk_id": i, "text": ch})

    print(f"[INFO] Total de chunks: {len(chunks)}")
    embs = embed_texts(chunks)  # numpy array (n, dim)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine (normalizado en embedder)
    index.add(embs)

    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"dim": int(dim), "chunks": mapping}, f, ensure_ascii=False, indent=2)
    print(f"[OK] Índice guardado en {INDEX_FILE} y metadatos en {META_FILE}")

if __name__ == "__main__":
    build_index()
