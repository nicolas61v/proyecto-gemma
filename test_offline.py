"""
Script para verificar que el modo offline funciona correctamente.
Simula desconexión de internet.
"""

import os
import sys

# Configurar UTF-8 en Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configurar modo offline ANTES de importar los modelos
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.expanduser('~/.cache/huggingface/hub')

print("="*70)
print("TEST OFFLINE - Verificar que la app funciona sin internet")
print("="*70)
print()

print("1. Verificando que los modelos están en caché...")
cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
if os.path.exists(cache_dir):
    models = os.listdir(cache_dir)
    models = [m for m in models if m.startswith('models--')]
    print(f"   Modelos encontrados en {cache_dir}:")
    for model in models:
        print(f"   - {model}")
    if len(models) == 0:
        print("   ADVERTENCIA: No hay modelos en caché")
        print("   Ejecuta: python descargar_modelos.py")
else:
    print(f"   ERROR: Directorio {cache_dir} no existe")
    sys.exit(1)

print()
print("2. Intentando cargar SentenceTransformer en modo offline...")
try:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder=os.path.expanduser('~/.cache/huggingface/hub'),
        local_files_only=True  # CRÍTICO
    )
    print("   [OK] SentenceTransformer cargado correctamente (OFFLINE)")

    # Probar que funciona
    test_emb = embed_model.encode(["test"])
    print(f"   [OK] Encoding funciona (dimension: {test_emb.shape})")
except Exception as e:
    print(f"   [ERROR] {str(e)[:150]}")
    print("   Solucion: Ejecuta 'python descargar_modelos.py' con internet")
    sys.exit(1)

print()
print("3. Intentando cargar Gemma 3 270M en modo offline...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    MODEL_NAME = "google/gemma-3-270m-it"
    print(f"   Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"   [OK] Tokenizer cargado")

    print(f"   Cargando modelo (esto tarda ~10 segundos)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    print(f"   [OK] Modelo Gemma cargado correctamente (OFFLINE)")
except Exception as e:
    print(f"   [ERROR] {str(e)[:150]}")
    print("   Solucion: Ejecuta 'python descargar_modelos.py' con internet")
    sys.exit(1)

print()
print("="*70)
print("SUCCESS: Todos los modelos funcionan en modo OFFLINE")
print("="*70)
print()
print("Puedes ejecutar la aplicación sin internet:")
print("  python gemma3_270m_chat.py")
print()
