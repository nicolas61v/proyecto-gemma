# Chat Local con Gemma 3 270M Instruct

Proyecto de Especialización en Small Language Models (SLM)
Universidad EAFIT, 2025

Autores: Felipe Castro Jaimes, Nicolás Vázquez, José Jiménez

---

## Descripción del Proyecto

Este es un chatbot local basado en Gemma 3 270M (modelo instruction-tuned de Google). La aplicación ofrece una interfaz web interactiva con capacidades de procesamiento de lenguaje natural, indexación de documentos y recuperación de información.

### Características Principales

- Chat interactivo mediante interfaz web (Gradio)
- RAG (Retrieval-Augmented Generation) con indexación local de documentos
- Soporte para múltiples formatos de entrada: TXT, MD, PDF
- Funcionamiento completamente local sin dependencias de APIs externas
- Parámetros configurables de temperatura y límite de tokens

---

## Requisitos Previos

- Python 3.8 o superior
- Token de Hugging Face (gratuito, disponible en https://huggingface.co/settings/tokens)
- Espacio en disco: mínimo 10 GB para el modelo
- Acceso a internet para descargar el modelo en la primera ejecución

---

## Instalación

### Opción A: Script Automático (Windows)

```bash
instalar_windows_gemma3.bat
```

Este script realiza automáticamente:
- Creación del entorno virtual
- Instalación de todas las dependencias
- Configuración inicial de la aplicación

### Opción B: Instalación Manual

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

### Configuración de Token de Hugging Face

Es necesario configurar el token para acceder al modelo:

```bash
huggingface-cli login
```

Se solicitará el token, el cual puede obtenerse en:
1. Visitando https://huggingface.co/settings/tokens
2. Creando un nuevo token con permisos de lectura
3. Pegando el token en la terminal cuando se solicite

---

## Ejecución

### En Windows

```bash
ejecutar_gemma3.bat
```

### En Linux o macOS

```bash
python gemma3_270m_chat.py
```

La aplicación se abrirá automáticamente en el navegador. Si no es así, acceder a:
```
http://127.0.0.1:7860
```

---

## Funcionalidades

### 1. Chat Interactivo

La interfaz permite realizar preguntas al modelo Gemma 3 270M con los siguientes controles:

- Temperatura: controla el nivel de aleatoridad (0.1 = más determinista, 0.9 = más creativo)
- Límite de tokens: define la longitud máxima de la respuesta
- Historial de conversación: mantiene las últimas interacciones

### 2. Recuperación Aumentada por Generación (RAG)

Permite que el modelo responda preguntas basadas en documentos cargados:

**Procedimiento:**

1. Colocar archivos en la carpeta `knowledge/` o usar la opción de carga en la interfaz
2. Presionar "Reconstruir Índice" para indexar los documentos
3. El sistema crea automáticamente un índice FAISS para búsqueda eficiente
4. Las respuestas incluirán contexto extraído de los documentos cargados

**Formatos soportados:**
- Archivos de texto (.txt)
- Markdown (.md)
- Documentos PDF (.pdf)

**Estructura de directorios:**

```
proyecto-gemma/
├── gemma3_270m_chat.py
├── requirements_gemma3.txt
├── knowledge/
│   ├── documento1.txt
│   ├── documento2.md
│   └── documento3.pdf
├── knowledge_index.faiss
├── knowledge_embeddings.npy
└── knowledge_metadata.json
```

---

## Solución de Problemas

### ModuleNotFoundError

Verificar que el entorno virtual está activado:

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### No authorization token provided

Configurar el token de Hugging Face:

```bash
huggingface-cli login
```

### CUDA out of memory

Reducir el límite máximo de tokens en la interfaz o ejecutar en CPU.

### Modelo no responde correctamente

Verificar que se está utilizando `google/gemma-3-270m-it` (instruction-tuned):
- Correcto: `gemma-3-270m-it` (sigue instrucciones)
- Incorrecto: `gemma-3-270m` (solo continúa texto)

---

## Estructura del Proyecto

```
proyecto-gemma/
├── gemma3_270m_chat.py           Aplicación principal
├── requirements_gemma3.txt       Dependencias Python
├── instalar_windows_gemma3.bat   Instalador automático
├── ejecutar_gemma3.bat           Ejecutor para Windows
├── .gitignore                    Configuración Git
├── README.md                     Este archivo
└── knowledge/                    Directorio para documentos
```

---

## Dependencias Principales

- torch: Framework de computación numérica y deep learning
- transformers: Modelos preentrenados de Hugging Face
- gradio: Framework para crear interfaces web
- sentence-transformers: Generación de embeddings para RAG
- faiss-cpu: Búsqueda eficiente de vectores
- pypdf: Lectura de archivos PDF

Véase `requirements_gemma3.txt` para la lista completa.

---

## Especificaciones Técnicas

| Componente | Especificación |
|-----------|---|
| Modelo | Gemma 3 270M Instruction-Tuned |
| Parámetros | 270 millones |
| Tokens de entrenamiento | 6 trillones |
| Tipo | Instruction-Tuned |
| Interfaz | Gradio |
| Motor RAG | FAISS + Sentence Transformers |
| Embeddings | all-MiniLM-L6-v2 (384 dimensiones) |

---

## Requisitos del Sistema

### Mínimos
- CPU moderna (Intel/AMD)
- 8 GB RAM
- 10 GB espacio en disco
- Python 3.8+

### Recomendados
- GPU NVIDIA con CUDA 11.8+
- 16 GB RAM
- 20 GB SSD
- Python 3.10+

### Óptimos
- GPU NVIDIA RTX 3060 o superior
- 32 GB RAM
- 100 GB SSD
- Windows 11 o Ubuntu 22.04+

---

## Uso Básico

### Ejemplo 1: Chat Simple

```
Usuario: Hola, explícame qué es un modelo de lenguaje
Bot: Un modelo de lenguaje es...
```

### Ejemplo 2: Preguntas Generales

```
Usuario: ¿Cuáles son las diferencias entre un transformer y un RNN?
Bot: Los transformers y RNNs son arquitecturas de red neuronal...
```

### Ejemplo 3: Uso de RAG

1. Cargar un documento con contenido académico
2. Hacer una pregunta relacionada
3. El modelo responde basándose en el contenido del documento

---

## Configuración Avanzada

### Cambiar Modelo

Editar `gemma3_270m_chat.py` (línea 40):

```python
MODEL_NAME = "google/gemma-3-270m-it"
```

### Cambiar Modelo de Embeddings para RAG

Editar `gemma3_270m_chat.py` (línea 47):

```python
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
```

### Optimizar para CPU

En `gemma3_270m_chat.py` (líneas 67-75), cambiar:

```python
torch_dtype=torch.float32  # En lugar de torch.float16 para CPU
```

---

## Notas Importantes

- El modelo se descargará automáticamente en la primera ejecución (aproximadamente 241 MB)
- El entorno virtual debe estar activado siempre antes de ejecutar la aplicación
- Es obligatorio configurar el token de Hugging Face para acceder al modelo
- Para mejores resultados con RAG, usar documentos estructurados y bien formateados

---

## Licencia

Este proyecto es parte del programa de especialización en Small Language Models de la Universidad EAFIT, 2025.

---

## Contacto y Soporte

Para problemas o consultas:

1. Revisar la sección de "Solución de Problemas"
2. Verificar que Python 3.8+ está instalado
3. Confirmar que el token de Hugging Face es válido
4. Revisar que el archivo `requirements_gemma3.txt` está presente
