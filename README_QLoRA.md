# QLoRA / RAG - Guía rápida (español)

Resumen:
- Este repositorio contiene un chat local con Gemma 3 270M (archivo `gemma3_270m_chat.py`).
- Aquí añadimos soporte para RAG (recuperación local con embeddings + FAISS) y un ejemplo de script para entrenar adaptadores LoRA (QLoRA).

1) Requisitos
- Se añadió/actualizó `requirements_gemma3.txt` con librerías adicionales:
  - `sentence-transformers`, `faiss-cpu` para RAG
  - `peft`, `bitsandbytes` para QLoRA

Nota importante sobre Windows:
- bitsandbytes y el entrenamiento en 4-bit funcionan mejor en Linux. Para usar QLoRA en Windows se recomienda:
  - Usar WSL2 con GPU passthrough (requiere drivers y configuración) o
  - Usar una máquina Linux con CUDA y GPUs NVIDIA.

2) Uso inmediato: RAG (memoria basada en archivos)
- Crea una carpeta `knowledge` junto al script y pon archivos de texto (.txt, .md) o jsonl con contenido que quieras que el agente use.
- Al iniciar `gemma3_270m_chat.py`, el script intentará leer la carpeta `knowledge` (si existen índices FAISS/embeddings los usa) y recuperará contextos relevantes para cada consulta.

3) Entrenar LoRA (QLoRA) - plantilla
- Archivo: `qlora_finetune.py` (plantilla)
- Formato de entrenamiento: archivo `train.jsonl` con líneas JSON: {"prompt":"...", "response":"..."}

Ejemplo de comando (PowerShell / WSL):

```powershell
# Instalar dependencias (PowerShell):
python -m pip install -r requirements_gemma3.txt

# Ejecutar entrenamiento (recomendado en WSL/Linux):
python qlora_finetune.py --train_file data/train.jsonl --output_dir lora_adapter --num_epochs 3 --batch_size 4
```

Después del entrenamiento, se generará la carpeta `lora_adapter`. El chat intenta cargar automáticamente esa carpeta al iniciarse.

4) Notas de rendimiento y recomendaciones
- Si no tienes GPU o tienes pocos recursos, usa RAG en lugar de fine-tuning; RAG permite que el modelo pequeño responda con información específica sin re-entrenarlo.
- Para cargas más grandes de conocimiento, preprocesa archivos en fragmentos y construye un índice FAISS (el script del chat hace una aproximación simple; puedes mejorarlo creando fragmentos y almacenando embeddings explícitamente).

5) Próximos pasos sugeridos
- Mejorar el pipeline de indexado (chunking + embeddings persistentes).
- Añadir un script para construir el índice FAISS (offline) y guardar mapeos archivo->fragmento.
- Si quieres, puedo:
  - Implementar el indexador offline y el mapeo fragmento->id
  - Preparar un dataset de entrenamiento a partir de carpetas de docs
  - Ajustar el `qlora_finetune.py` para un entrenamiento más estable

