import os, re, json, pathlib
from typing import List, Dict, Tuple
from pypdf import PdfReader

def load_text_from_file(path: str) -> str:
    path = str(path)
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(texts).strip()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def clean_text(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # cortar limpio por oraciones si es posible
        last_dot = chunk.rfind(". ")
        if last_dot > chunk_size * 0.6:
            end = start + last_dot + 1
            chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, 0)
        if start >= len(text):
            break
    return chunks

def iter_docs(data_dir: str) -> List[Tuple[str, str]]:
    """Return list of (path, content) for files in data_dir."""
    paths = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn.lower().endswith((".txt", ".md", ".pdf")):
                p = os.path.join(root, fn)
                try:
                    txt = load_text_from_file(p)
                    if txt and txt.strip():
                        paths.append((p, txt))
                except Exception as e:
                    print(f"[WARN] No se pudo leer {p}: {e}")
    return paths
