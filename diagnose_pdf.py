"""
Script de diagnóstico para PDFs
"""
from pypdf import PdfReader
import pathlib
import os

pdf_path = 'knowledge/ai-article.pdf'  # Cambiar por la ruta del PDF a diagnosticar

print("=" * 60)
print("DIAGNOSTICO DE EXTRACCION DE PDF")
print("=" * 60)
print()

try:
    print(f"PDF: {pdf_path}")
    print(f"Existe: {pathlib.Path(pdf_path).exists()}")

    file_size = pathlib.Path(pdf_path).stat().st_size / (1024*1024)
    print(f"Tamaño: {file_size:.2f} MB")
    print()

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"Total de páginas: {num_pages}")
    print()

    # Análisis de contenido
    print("Analizando contenido...")
    total_chars = 0
    empty_pages = 0

    for i in range(num_pages):
        text = reader.pages[i].extract_text()
        text_len = len(text) if text else 0
        total_chars += text_len

        if text_len < 10:
            empty_pages += 1
            print(f"  Página {i+1}: {text_len} caracteres [PROBLEMA]")
        else:
            print(f"  Página {i+1}: {text_len} caracteres [OK]")

    print()
    print(f"Total de caracteres extraidos: {total_chars}")
    print(f"Páginas vacías/problemáticas: {empty_pages} de {num_pages}")
    print()

    if total_chars > 0:
        print("RESULTADO: PDF se puede procesar")
        print(f"Tamaño aproximado de índice FAISS: ~{total_chars // 500 * 50} MB")
    else:
        print("RESULTADO: PDF posiblemente escaneado (imágenes), se necesita OCR")

except Exception as e:
    print(f"ERROR: {type(e).__name__}")
    print(f"Mensaje: {str(e)}")
