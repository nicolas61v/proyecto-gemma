"""
Script mejorado de diagnóstico RAG
Analiza extracción de PDF, construcción de índice y búsquedas
"""
import os
import sys
import json
import pathlib
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# Configurar encoding UTF-8 en Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("DIAGNOSTICO COMPLETO DE RAG")
print("=" * 80)
print()

# Configuración
KNOWLEDGE_DIR = "knowledge"
FAISS_INDEX_PATH = "knowledge_index.faiss"
EMBEDDINGS_PATH = "knowledge_embeddings.npy"
METADATA_PATH = "knowledge_metadata.json"

# 1. ANÁLISIS DE EXTRACCIÓN DE PDF
print("1. ANÁLISIS DE EXTRACCIÓN DE PDF")
print("-" * 80)

p = pathlib.Path(KNOWLEDGE_DIR)
if p.exists():
    pdfs = list(p.glob("*.pdf"))
    print(f"PDFs encontrados: {len(pdfs)}")

    for pdf_path in pdfs:
        print(f"\nPDF: {pdf_path.name}")
        file_size = pdf_path.stat().st_size / (1024*1024)
        print(f"  Tamaño: {file_size:.2f} MB")

        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)
            print(f"  Páginas: {num_pages}")

            total_chars = 0
            empty_pages = 0
            sample_text = ""

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text_len = len(text)
                total_chars += text_len

                if text_len < 50:
                    empty_pages += 1
                    status = "[VACIA]"
                else:
                    status = "[OK]"
                    if not sample_text:
                        sample_text = text[:200]

                print(f"    Pagina {i+1}: {text_len:5d} caracteres {status}")

            print(f"\n  Total de caracteres: {total_chars:,}")
            print(f"  Promedio por página: {total_chars // num_pages:,}")
            print(f"  Páginas problemáticas: {empty_pages}/{num_pages}")

            if sample_text:
                print(f"\n  Sample de texto extraido:")
                print(f"    {sample_text[:150]}...")

            if total_chars < 1000:
                print("\n  [!] PROBLEMA: Muy poco texto extraido. Revisar formato PDF.")
            elif empty_pages > num_pages * 0.3:
                print("\n  [!] PROBLEMA: Muchas paginas vacias. Posible PDF escaneado.")
            else:
                print("\n  [OK] PDF se extrae correctamente")

        except Exception as e:
            print(f"  [ERROR] al leer PDF: {e}")

else:
    print(f"[!] Directorio '{KNOWLEDGE_DIR}' no existe")

print()
print()

# 2. ANÁLISIS DE ÍNDICE FAISS
print("2. ANÁLISIS DE ÍNDICE FAISS")
print("-" * 80)

if os.path.exists(FAISS_INDEX_PATH):
    print(f"[OK] Indice FAISS encontrado: {FAISS_INDEX_PATH}")

    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"  Dimensiones: {index.d}")
    print(f"  Numero de vectores: {index.ntotal}")

    # Cargar embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        embs = np.load(EMBEDDINGS_PATH)
        print(f"\n[OK] Embeddings encontrados: {EMBEDDINGS_PATH}")
        print(f"  Forma: {embs.shape}")
        print(f"  Dtype: {embs.dtype}")
        print(f"  Rango valores: [{embs.min():.4f}, {embs.max():.4f}]")
    else:
        print(f"\n[ERROR] Embeddings NO encontrados: {EMBEDDINGS_PATH}")

    # Cargar metadata
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"\n[OK] Metadata encontrada: {METADATA_PATH}")
        print(f"  Total de chunks: {len(metadata)}")

        if metadata:
            print(f"\n  Primeros 3 chunks:")
            for i in range(min(3, len(metadata))):
                m = metadata[i]
                text_preview = m.get('text', '')[:80]
                source = m.get('source', 'unknown')
                print(f"    [{i}] {source}: {text_preview}...")

            print(f"\n  Últimos 3 chunks:")
            for i in range(max(0, len(metadata)-3), len(metadata)):
                m = metadata[i]
                text_preview = m.get('text', '')[:80]
                source = m.get('source', 'unknown')
                print(f"    [{i}] {source}: {text_preview}...")
    else:
        print(f"\n[ERROR] Metadata NO encontrada: {METADATA_PATH}")
else:
    print(f"[ERROR] Indice FAISS NO encontrado: {FAISS_INDEX_PATH}")

print()
print()

# 3. PRUEBAS DE BÚSQUEDA RAG
print("3. PRUEBAS DE BÚSQUEDA RAG")
print("-" * 80)

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
    print("Cargando modelo de embeddings...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Pruebas
    test_queries = [
        "¿Qué es la inteligencia artificial?",
        "¿Qué es la IA?",
        "inteligencia artificial",
        "IA",
        "historia de la IA",
        "definición de inteligencia artificial",
    ]

    thresholds = [2.5, 2.0, 3.0, 4.0]

    print(f"\nProbando {len(test_queries)} queries con diferentes thresholds:\n")

    for query in test_queries:
        print(f"QUERY: '{query}'")
        q_emb = embed_model.encode([query])
        D, I = index.search(np.array(q_emb).astype('float32'), k=5)

        print(f"  Top-5 resultados (sin threshold):")
        for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
            if idx < len(metadata):
                text_preview = metadata[idx]['text'][:60]
                source = metadata[idx]['source']
                print(f"    {rank}. dist={dist:.4f} [{source}] {text_preview}...")

        print(f"\n  Resultados por threshold:")
        for threshold in thresholds:
            matches = sum(1 for d in D[0] if d < threshold)
            if matches > 0:
                status = "[OK]"
            else:
                status = "[FAIL]"
            print(f"    {threshold}: {matches} matches {status}")
        print()

else:
    print("[ERROR] No se puede hacer busquedas: indice no construido")

print()
print("=" * 80)
print("RESUMEN DE DIAGNÓSTICO")
print("=" * 80)

checks = [
    ("PDF existe y contiene texto", os.path.exists(FAISS_INDEX_PATH)),
    ("Índice FAISS construido", os.path.exists(FAISS_INDEX_PATH)),
    ("Embeddings guardados", os.path.exists(EMBEDDINGS_PATH)),
    ("Metadata guardada", os.path.exists(METADATA_PATH)),
]

all_ok = True
for check, status in checks:
    symbol = "[OK]" if status else "[FAIL]"
    print(f"{symbol} {check}")
    if not status:
        all_ok = False

print()
if all_ok:
    print("[OK] Sistema RAG esta completo. Si no responde correctamente:")
    print("  - Aumentar threshold de relevancia en chat_with_gemma()")
    print("  - Usar modelo de embeddings mas potente")
    print("  - Revisar chunking del texto")
else:
    print("[ERROR] Sistema RAG incompleto. Pasos a seguir:")
    print("  1. Asegurate de tener PDFs en la carpeta 'knowledge/'")
    print("  2. Presiona boton 'Indexar' en la UI de Gradio")
    print("  3. Ejecuta este script nuevamente para verificar")
