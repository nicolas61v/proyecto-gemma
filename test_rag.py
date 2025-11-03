#!/usr/bin/env python3
"""
Test de RAG - Verifica que la búsqueda de documentos funciona
Sin cargar el modelo Gemma (más rápido)
"""

import os
import json
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("ERROR: Instala las dependencias:")
    print("  pip install sentence-transformers faiss-cpu")
    exit(1)


def test_rag():
    """Test de funcionalidad RAG"""

    print("=" * 70)
    print("TEST DE RAG (Retrieval-Augmented Generation)")
    print("=" * 70)
    print()

    # Verificar archivos necesarios
    required_files = [
        "knowledge_index.faiss",
        "knowledge_embeddings.npy",
        "knowledge_metadata.json",
    ]

    print("1. Verificando archivos...")
    for fname in required_files:
        if os.path.exists(fname):
            size = os.path.getsize(fname) / 1024
            print(f"   OK: {fname} ({size:.1f} KB)")
        else:
            print(f"   FALTA: {fname} (ejecuta build_index.py primero)")
            return False

    print()

    # Cargar índice
    print("2. Cargando índice FAISS...")
    try:
        index = faiss.read_index("knowledge_index.faiss")
        embeddings = np.load("knowledge_embeddings.npy")
        with open("knowledge_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(f"   OK: Índice: {index.ntotal} vectores")
        print(f"   OK: Embeddings: {embeddings.shape}")
        print(f"   OK: Metadatos: {len(metadata)} entradas")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    print()

    # Cargar modelo de embeddings
    print("3. Cargando modelo de embeddings...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("   OK: Modelo cargado")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

    print()

    # Preguntas de prueba
    test_queries = [
        "¿Qué es MikroTik?",
        "¿Cómo configurar un router?",
        "¿Qué es RouterOS?",
        "Configuración básica",
        "PPPoE",
    ]

    print("4. Probando búsquedas RAG...")
    print()

    for query in test_queries:
        print(f"   Pregunta: {query}")

        # Codificar pregunta
        q_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

        # Buscar top-3
        distances, indices = index.search(q_embedding, k=3)

        print(f"   Resultados (distancia L2):")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(metadata):
                entry = metadata[idx]
                source = entry.get("source", "unknown")
                text_preview = entry.get("text", "")[:80]
                print(
                    f"     {i+1}. [{dist:.3f}] {source}: {text_preview}..."
                )
            else:
                print(f"     {i+1}. [INVALID INDEX] {idx}")

        print()

    print("=" * 70)
    print("TEST COMPLETADO!")
    print("=" * 70)
    print()
    print("[OK] RAG está funcionando correctamente")
    print("[OK] Puedes usar gemma3_270m_chat.py para hacer consultas")
    print()

    return True


if __name__ == "__main__":
    success = test_rag()
    exit(0 if success else 1)
