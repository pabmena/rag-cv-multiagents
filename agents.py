
import os, json, pathlib
from typing import List, Dict, Any, Tuple
import faiss
from dotenv import load_dotenv
from rag import get_embedder, embed_texts, generate_answer  # reuse generation + embedder
from utils import iter_docs, chunk_text

load_dotenv()

TOP_K = int(os.getenv("TOP_K", "4"))
STORAGE_DIR = pathlib.Path("storage")

class PersonAgent:
    def __init__(self, name: str):
        self.name = name
        self.index_dir = STORAGE_DIR / name
        self.index_file = self.index_dir / "index.faiss"
        self.meta_file = self.index_dir / "meta.json"
        self._index = None
        self._meta = None
        self._embedder = None

    # -------- Retrieval --------
    def _load(self):
        if self._index is None:
            if not self.index_file.exists() or not self.meta_file.exists():
                raise RuntimeError(f"No existe índice para {self.name}. Ejecutá: python ingest_agents.py")
            self._index = faiss.read_index(str(self.index_file))
        if self._meta is None:
            self._meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._index, self._meta, self._embedder

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        index, meta, embedder = self._load()
        top_k = top_k or TOP_K
        q_emb = embedder.encode([query], normalize_embeddings=True)
        D, I = index.search(q_emb, top_k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            info = meta["chunks"][idx]
            hits.append({
                "score": float(score),
                "text": info["text"],
                "source": info["source"],
                "chunk_id": int(idx),
                "person": self.name,
            })
        return hits

    # -------- Generation --------
    def build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, c in enumerate(contexts, start=1):
            blocks.append(f"[{i}] ({c['source']})\n{c['text']}")
        context_block = "\n\n".join(blocks) if blocks else "(sin contexto)"
        system = f"""Actuás como un asistente RAG especializado en el CV de {self.name}.
- Respondé SOLO con base al contexto citado.
- Si no hay evidencia suficiente, decílo explícitamente.
- Usá español neutro, oraciones breves y viñetas cuando ayude.
- Agregá una sección 'Fuentes' con los índices usados por persona.
"""
        user = f"""Consulta sobre {self.name}: {query}

Contexto:
{context_block}
"""
        return system.strip(), user.strip()

    def answer(self, query: str, top_k: int = None) -> Tuple[str, List[Dict[str, Any]]]:
        ctxs = self.retrieve(query, top_k=top_k)
        system, user = self.build_prompt(query, ctxs)
        prompt = f"[system]\n{system}\n\n[user]\n{user}"
        answer = generate_answer(prompt)
        return answer, ctxs


def load_people_config(path: str = "config/people.json") -> Dict[str, Any]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError("Falta config/people.json")
    return json.loads(p.read_text(encoding="utf-8"))

def list_people() -> List[str]:
    cfg = load_people_config()
    return [p["name"] for p in cfg.get("people", [])]

def get_default_student() -> str:
    cfg = load_people_config()
    return cfg.get("default_student") or (cfg.get("people", [{}])[0].get("name"))

def resolve_people_from_query(query: str) -> List[str]:
    q = query.lower()
    cfg = load_people_config()
    selected = []
    for person in cfg.get("people", []):
        name = person["name"]
        aliases = [name.lower()] + [a.lower() for a in person.get("aliases", [])]
        if any(a in q for a in aliases):
            selected.append(name)
    if not selected:
        selected = [get_default_student()]
    return list(dict.fromkeys(selected))  # unique, preserve order
