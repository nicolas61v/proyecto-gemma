#!/usr/bin/env python3
"""
Script mejorado para construir índice FAISS
- Más robusto con manejo de errores
- Muestra progreso detallado
- Optimizado para bajo consumo de RAM
- Soporta archivos grandes
"""

import os
import pathlib
import json
import sys
from typing import List, Tuple
import numpy as np

# Importaciones condicionales
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None


def extract_text_from_file(path: str) -> Tuple[str, str]:
    """
    Extrae texto de PDF, TXT o MD.
    Retorna: (texto, nombre_archivo)
    """
    p = pathlib.Path(path)
    if not p.exists():
        return "", p.name

    try:
        if p.suffix.lower() == ".pdf":
            if PdfReader is None:
                print(f"  SKIP: {p.name} (pypdf no instalado)")
                return "", p.name

            text = []
            try:
                reader = PdfReader(str(p))
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        text.append(page_text)
                    except Exception as e:
                        print(f"    WARN: Error en página {i+1}: {e}")
                        continue
            except Exception as e:
                print(f"  ERROR: No se pudo leer {p.name}: {e}")
                return "", p.name

            full_text = "\n".join(text)
            return full_text, p.name

        else:
            # TXT, MD, etc.
            try:
                return p.read_text(encoding="utf-8"), p.name
            except UnicodeDecodeError:
                try:
                    return p.read_text(encoding="latin-1"), p.name
                except Exception as e:
                    print(f"  ERROR: No se pudo leer {p.name}: {e}")
                    return "", p.name
    except Exception as e:
        print(f"  ERROR: {p.name}: {e}")
        return "", p.name


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Divide texto en chunks por palabras con solapamiento."""
    if not text:
        return []

    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


def build_index(
    knowledge_dir: str = "knowledge",
    chunk_size: int = 500,
    overlap: int = 50,
    embedding_model: str = "all-MiniLM-L6-v2",
):
    """Construye índice FAISS desde documentos."""

    if SentenceTransformer is None or faiss is None:
        print("ERROR: Instala sentence-transformers y faiss-cpu:")
        print("  pip install sentence-transformers faiss-cpu")
        return False

    print("=" * 70)
    print("CONSTRUIR ÍNDICE FAISS")
    print("=" * 70)
    print()

    # 1. Explorar archivos
    print(f"1. Explorando directorio: {knowledge_dir}")
    k_path = pathlib.Path(knowledge_dir)
    if not k_path.exists():
        print(f"   CREAR: Creando directorio {knowledge_dir}/")
        k_path.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in k_path.glob("**/*") if f.is_file()])
    print(f"   ENCONTRADO: {len(files)} archivos")
    print()

    if not files:
        print("   WARN: No hay archivos para indexar")
        return False

    # 2. Extraer texto
    print("2. Extrayendo texto de archivos...")
    all_chunks = []
    metadata = []
    chunk_id = 0

    for file_path in files:
        if file_path.suffix.lower() not in [".pdf", ".txt", ".md"]:
            continue

        print(f"   -> {file_path.name}...", end=" ", flush=True)

        text, filename = extract_text_from_file(str(file_path))

        if not text:
            print("SKIP (vacío)")
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"OK ({len(chunks)} chunks, {len(text)} caracteres)")

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append(
                {
                    "id": chunk_id,
                    "source": filename,
                    "text": chunk[:2000],  # Limita para JSON
                }
            )
            chunk_id += 1

    print()
    if not all_chunks:
        print("ERROR: No se extrajo texto de ningún archivo")
        return False

    print(f"   TOTAL: {len(all_chunks)} chunks")
    print()

    # 3. Generar embeddings
    print("3. Generando embeddings...")
    print(f"   Modelo: {embedding_model}")
    print("   (Primera vez descarga el modelo, ~100MB)")

    try:
        embed_model = SentenceTransformer(embedding_model)
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    print(f"   Codificando {len(all_chunks)} chunks...", flush=True)
    try:
        embeddings = embed_model.encode(
            all_chunks, convert_to_numpy=True, show_progress_bar=True
        )
        embeddings = embeddings.astype("float32")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    print(f"   Dimensión: {embeddings.shape[1]}")
    print()

    # 4. Crear índice FAISS
    print("4. Construyendo índice FAISS...")
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        print(f"   OK: {index.ntotal} vectores en índice")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    print()

    # 5. Guardar archivos
    print("5. Guardando archivos...")
    try:
        # Índice
        faiss_path = "knowledge_index.faiss"
        faiss.write_index(index, faiss_path)
        print(f"   OK: {faiss_path}")

        # Embeddings (numpy)
        embed_path = "knowledge_embeddings.npy"
        np.save(embed_path, embeddings)
        size_mb = embeddings.nbytes / (1024 * 1024)
        print(f"   OK: {embed_path} ({size_mb:.1f} MB)")

        # Metadata (JSON)
        meta_path = "knowledge_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"   OK: {meta_path}")

    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    print()
    print("=" * 70)
    print("ÍNDICE CONSTRUIDO EXITOSAMENTE!")
    print("=" * 70)
    print()
    print(f"Resumen:")
    print(f"  • Archivos procesados: {len(files)}")
    print(f"  • Chunks generados: {len(all_chunks)}")
    print(f"  • Dimensión embeddings: {embeddings.shape[1]}")
    print(f"  • Tamaño índice: ~{embeddings.nbytes / (1024*1024):.1f} MB")
    print()
    print(f"Archivos generados:")
    print(f"  1. knowledge_index.faiss")
    print(f"  2. knowledge_embeddings.npy")
    print(f"  3. knowledge_metadata.json")
    print()
    print("Ahora puedes usar gemma3_270m_chat.py para consultar con RAG!")
    print()

    return True


if __name__ == "__main__":
    success = build_index()
    sys.exit(0 if success else 1)
