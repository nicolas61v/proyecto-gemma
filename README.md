# 🤖 Chat Local con Gemma 3 270M Instruct

**Proyecto de Especialización en Small Language Models (SLM)**
Universidad EAFIT, 2025

Autores: Felipe Castro Jaimes, Nicolás Vázquez, José Jiménez

---

## 📋 Descripción del Proyecto

Este es un **chatbot local e inteligente** basado en Gemma 3 270M (modelo instruction-tuned de Google). La aplicación ofrece:

✅ **Chat interactivo** con interfaz web (Gradio)
✅ **RAG (Retrieval-Augmented Generation)** - indexación local de documentos
✅ **Soporte para fine-tuning** con QLora
✅ **Múltiples formatos** - TXT, MD, PDF
✅ **Totalmente local** - Sin dependencias de APIs externas

---

## 🚀 Inicio Rápido (Windows)

### 1️⃣ Requisitos Previos

- **Python 3.8+** ([descargar](https://www.python.org/downloads/))
- **Token de Hugging Face** (gratuito, [registrarse aquí](https://huggingface.co/))
- **~10 GB de espacio en disco** (para descargar el modelo)

### 2️⃣ Instalación

#### Opción A: Script automático (recomendado en Windows)

```bash
instalar_windows_gemma3.bat
```

Este script:
- ✅ Crea entorno virtual
- ✅ Instala todas las dependencias
- ✅ Configura la aplicación

#### Opción B: Instalación manual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Activar entorno (Linux/Mac)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements_gemma3.txt
```

### 3️⃣ Configurar Token de Hugging Face

Necesitas acceso al modelo de Google en Hugging Face:

```bash
huggingface-cli login
# Pega tu token cuando se te pida
```

**¿Dónde conseguir el token?**
1. Ve a https://huggingface.co/settings/tokens
2. Crea un nuevo token (read)
3. Pegalo en la terminal
4. u usa esta REDACTED

### 4️⃣ Ejecutar la Aplicación

#### Windows:
```bash
ejecutar_gemma3.bat
```

#### Linux/Mac:
```bash
python gemma3_270m_chat.py
```

**Resultado esperado:**
```
==============================================================
🚀 CHAT LOCAL CON GEMMA 3 270M INSTRUCT
   Proyecto SLM - Universidad EAFIT 2025
==============================================================
📱 Dispositivo: CUDA (o CPU)
🔧 Modelo: google/gemma-3-270m-it
✅ Usando versión INSTRUCTION-TUNED

⏳ Cargando Gemma 3 270M Instruct...
   (Primera vez descargará ~241MB)

✅ ¡Gemma 3 270M Instruct cargado exitosamente!

🌐 Abriendo interfaz web en http://127.0.0.1:7860
```

Luego abre en tu navegador: **http://127.0.0.1:7860**

---

## 💡 Características Principales

### 1. Chat Interactivo
- Respuestas coherentes y contextuales
- Historial de conversación (últimas 3 interacciones)
- Controles de temperatura y límite de tokens

### 2. RAG (Memoria con Documentos)
Permite que el modelo responda preguntas basadas en tus documentos:

**Pasos:**
1. Coloca archivos en la carpeta `knowledge/` (TXT, MD, PDF)
2. O sube archivos directamente en la interfaz
3. El sistema indexa automáticamente con FAISS
4. Las respuestas incluyen contexto de tus documentos

**Archivos soportados:**
- `.txt` - Archivos de texto
- `.md` - Markdown
- `.pdf` - Documentos PDF

**Estructura de directorios:**
```
proyecto-gemma/
├── knowledge/              # Coloca tus documentos aquí
│   ├── documento1.txt
│   ├── documento2.md
│   └── documento3.pdf
├── knowledge_index.faiss   # Índice (se crea automáticamente)
├── knowledge_embeddings.npy
└── knowledge_metadata.json
```

### 3. Fine-tuning con QLora
Entrena el modelo con tus propios datos:

**Preparar datos:**
Crea un archivo `train.jsonl`:
```json
{"prompt": "¿Qué es IA?", "response": "La IA es..."}
{"prompt": "¿Cómo funciona?", "response": "Funciona mediante..."}
```

**Entrenar** (recomendado en Linux/WSL2):
```bash
python qlora_finetune.py --train_file train.jsonl --output_dir lora_adapter --num_epochs 3
```

El modelo cargará automáticamente el adaptador LoRA si existe.

---

## ⚙️ Configuración Avanzada

### Ajustar Parámetros del Modelo

Edita `gemma3_270m_chat.py`:

```python
# Línea 40 - Cambiar modelo
MODEL_NAME = "google/gemma-3-270m-it"  # Usa este version!

# Línea 47 - Modelo de embeddings para RAG
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Línea 41 - Dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Optimizar para CPU
Si no tienes GPU, añade en `gemma3_270m_chat.py` (línea 67-75):

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Cambia float16 → float32 para CPU
    device_map=None,
    low_cpu_mem_usage=True
)
```

---

## 🐛 Solución de Problemas

### ❌ "ModuleNotFoundError"
**Solución:** Asegúrate de activar el entorno virtual
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### ❌ "No authorization token provided"
**Solución:** Configura tu token de Hugging Face
```bash
huggingface-cli login
```

### ❌ CUDA out of memory
**Solución:** Reduce los tokens máximos en la interfaz o usa CPU

### ❌ Respuestas extrañas
**Importante:** Asegúrate de usar `google/gemma-3-270m-it` (instruction-tuned)
- ✅ `gemma-3-270m-it` - Sigue instrucciones correctamente
- ❌ `gemma-3-270m` - Solo continúa texto, no sigue instrucciones

---

## 📁 Estructura del Proyecto

```
proyecto-gemma/
├── gemma3_270m_chat.py           # 🎯 Aplicación principal
├── qlora_finetune.py             # Entrenamiento con QLora
├── index_knowledge.py            # Construcción de índice FAISS
├── requirements_gemma3.txt       # Dependencias Python
├── instalar_windows_gemma3.bat   # Instalador Windows
├── ejecutar_gemma3.bat           # Ejecutor Windows
├── .gitignore                    # Git ignore
├── README.md                     # Este archivo
├── README_QLoRA.md               # Guía detallada de QLora
└── knowledge/                    # Documentos (tú creas esta carpeta)
```

---

## 📦 Dependencias

**Núcleo:**
- `torch` - Framework de deep learning
- `transformers` - Modelos de Hugging Face
- `gradio` - Interfaz web

**Opcional pero recomendado:**
- `sentence-transformers` - Embeddings para RAG
- `faiss-cpu` - Búsqueda de similaridad
- `peft` - QLora fine-tuning
- `bitsandbytes` - Optimizaciones de entrenamiento

**Conversión de archivos:**
- `pypdf` - Lectura de PDFs

---

## 🎓 Primeros Pasos

### Prueba 1: Chat Simple
```
Usuario: Hola, ¿cómo estás?
Bot: Hola! Estoy bien, gracias por preguntar...
```

### Prueba 2: Preguntas de Conocimiento
```
Usuario: ¿Qué es un transformer en IA?
Bot: Un transformer es una arquitectura de red neuronal...
```

### Prueba 3: Usar RAG
1. Sube un PDF o TXT con información
2. Pregunta algo relacionado
3. El modelo responderá basándose en tu documento

---

## 🔍 Información del Sistema

| Componente | Detalles |
|-----------|----------|
| **Modelo** | Gemma 3 270M Instruction-Tuned |
| **Parámetros** | 270 millones |
| **Entrenamiento** | 6 trillones de tokens |
| **Tipo** | Instruction-Tuned (sigue instrucciones) |
| **Interfaz** | Gradio (web) |
| **RAG** | FAISS + Sentence Transformers |
| **Entrenamiento** | QLora (4-bit) |

---

## 💻 Requisitos del Sistema

### Mínimos
- CPU moderna (Intel/AMD)
- 8 GB RAM
- 10 GB disco
- Python 3.8+

### Recomendados
- **GPU NVIDIA** (CUDA 11.8+)
- 16 GB RAM
- 20 GB SSD
- Python 3.10+

### Óptimos
- **GPU NVIDIA RTX 3060+**
- 32 GB RAM
- 100 GB SSD
- Windows 11 o Ubuntu 22.04+

---

## 🤝 Contribuciones y Mejoras

Posibles mejoras futuras:
- [ ] Interfaz mejorada (FastAPI)
- [ ] Soporte para más modelos SLM
- [ ] Indexador FAISS offline más eficiente
- [ ] Dashboard de estadísticas
- [ ] API REST

---

## 📝 Licencia

Este proyecto es parte del programa de especialización en SLM de EAFIT 2025.

---

## 📞 Soporte

Para problemas o preguntas:
1. Revisa la sección de **Solución de Problemas**
2. Lee `README_QLoRA.md` para aspectos avanzados
3. Verifica que tienes el token de Hugging Face correcto

---

## 🚀 Próximos Pasos

1. ✅ Instala la aplicación
2. ✅ Prueba el chat
3. ✅ Sube documentos para RAG
4. ✅ (Opcional) Entrena con QLora
5. ✅ ¡Sube a GitHub!

**¡Disfruta tu chatbot local! 🎉**
"# proyecto-gemma" 
