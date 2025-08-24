
import os, json, pathlib
from typing import Dict, Any, List
import faiss
from rag import get_embedder, embed_texts
from utils import iter_docs, chunk_text

DATA_DIR = pathlib.Path("data")
STORAGE_DIR = pathlib.Path("storage")

def build_index_for_person(person: str) -> Dict[str, Any]:
    person_dir = DATA_DIR / person
    if not person_dir.exists():
        print(f"[WARN] No existe data para {person}: {person_dir}")
        return {}

    docs = iter_docs(str(person_dir))
    if not docs:
        print(f"[WARN] {person} no tiene documentos en {person_dir}")
        return {}

    # Build chunks
    chunks, mapping = [], []
    for path, text in docs:
        for ch in chunk_text(text):
            mapping.append({"text": ch, "source": str(path)})
            chunks.append(ch)
    print(f"[INFO] {person}: {len(chunks)} chunks")

    # Embed + index
    embs = embed_texts(chunks)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # Persist
    out_dir = STORAGE_DIR / person
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "meta.json").write_text(json.dumps({"dim": int(dim), "chunks": mapping}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Índice de {person} en {out_dir}")
    return {"person": person, "chunks": len(chunks)}

def build_all():
    people = [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
    if not people:
        print("[ERROR] No hay carpetas en /data. Crea /data/<Persona>/ y agrega sus CVs (PDF/TXT/MD).")
        return
    print(f"[INFO] Personas detectadas: {people}")
    for p in people:
        build_index_for_person(p)

if __name__ == "__main__":
    build_all()
