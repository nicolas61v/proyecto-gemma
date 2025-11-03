"""
Script ligero para convertir PDFs/txt/md a texto, fragmentar y construir índice FAISS.
No carga el modelo de Gemma (evita descargar pesos grandes al importar).

Uso:
  C:/.../.env/Scripts/python.exe index_knowledge.py

Genera:
  - knowledge_index.faiss
  - knowledge_embeddings.npy
  - knowledge_metadata.json

"""
import os
import pathlib
import json
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except Exception as e:
    print("Faltan dependencias: ", e)
    raise


BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "knowledge_index.faiss")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "knowledge_embeddings.npy")
METADATA_PATH = os.path.join(BASE_DIR, "knowledge_metadata.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def extract_text(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".pdf":
        if PdfReader is None:
            print("pypdf no está instalado, no se puede leer PDF:", path)
            return ""
        try:
            reader = PdfReader(str(path))
            pages = []
            for p in reader.pages:
                pages.append(p.extract_text() or "")
            return "\n".join(pages)
        except Exception as e:
            print("Error leyendo PDF", path, e)
            return ""
    else:
        try:
            return path.read_text(encoding='utf-8')
        except Exception:
            try:
                return path.read_text(encoding='latin-1')
            except Exception:
                return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def main():
    p = pathlib.Path(KNOWLEDGE_DIR)
    if not p.exists():
        print("No existe la carpeta knowledge/. Crea y añade archivos (txt, md, pdf) y vuelve a ejecutar.")
        return

    docs = []
    metadata = []
    for fp in sorted(p.glob("**/*")):
        if fp.is_file():
            text = extract_text(fp)
            if not text:
                continue
            chunks = chunk_text(text)
            for c in chunks:
                metadata.append({
                    "id": len(metadata),
                    "source": str(fp.name),
                    "text": c[:2000]
                })
                docs.append(c)

    if not docs:
        print("No se encontraron documentos o no se pudo extraer texto.")
        return

    print(f"Generando embeddings con {EMBED_MODEL_NAME} para {len(docs)} fragmentos...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embs = embed_model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    embs = embs.astype('float32')

    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embs)
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Índice FAISS creado:", FAISS_INDEX_PATH)
    print("Metadata guardada:", METADATA_PATH)


if __name__ == '__main__':
    main()
