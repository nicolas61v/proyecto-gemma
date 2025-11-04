"""
Script para descargar los modelos por adelantado
Ejecutar SOLO UNA VEZ cuando tengas internet
Esto permite usar la app sin internet después
"""

import os
import sys

print("="*80)
print("DESCARGADOR DE MODELOS - Ejecutar UNA SOLA VEZ con Internet")
print("="*80)
print()

# Configuración
MODEL_NAME = "google/gemma-3-270m-it"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

print("Este script descargará los modelos necesarios para usar la app sin internet.")
print()
print(f"Modelo principal: {MODEL_NAME} (~241 MB)")
print(f"Modelo embeddings: {EMBED_MODEL_NAME} (~100 MB)")
print()
print("REQUIERE: Token válido de Hugging Face")
print("Consigue uno en: https://huggingface.co/settings/tokens")
print()

try:
    # Descargar modelo principal
    print("1. Descargando modelo principal (esto puede tardar 2-5 minutos)...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print("   - Descargando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("   - Tokenizador descargado correctamente")

    print("   - Descargando modelo (esto es lo más lento)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    print("   - Modelo descargado correctamente")
    print()

    # Descargar modelo de embeddings
    print("2. Descargando modelo de embeddings (2-3 minutos)...")
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    print("   - Modelo de embeddings descargado correctamente")
    print()

    print("="*80)
    print("DESCARGA COMPLETADA EXITOSAMENTE!")
    print("="*80)
    print()
    print("Ahora puedes usar la aplicación sin internet:")
    print("  - ejecutar_gemma3.bat (en Windows)")
    print("  - python gemma3_270m_chat.py (en Linux/Mac)")
    print()
    print("Los modelos están cacheados en:")
    print(f"  - ~/.cache/huggingface/hub/")
    print()
    print("Si desconectas internet, la app seguirá funcionando.")
    print()

except Exception as e:
    print()
    print("="*80)
    print("ERROR DURANTE LA DESCARGA")
    print("="*80)
    print(f"Error: {str(e)}")
    print()
    print("Posibles soluciones:")
    print("1. Verifica que tienes conexión a internet")
    print("2. Ejecuta: huggingface-cli login")
    print("3. Pega tu token cuando se solicite")
    print("4. Vuelve a ejecutar este script")
    print()
    sys.exit(1)
