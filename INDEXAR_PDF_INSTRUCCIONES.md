# Cómo Indexar tu PDF en el Sistema RAG

## Problema Resuelto

Antes: Solo podías indexar PDFs subiendo archivos nuevos mediante el botón "Subir e Indexar"

Ahora: Tienes **DOS opciones** para indexar:

1. **"Subir e Indexar"** - Sube un PDF nuevo Y lo indexa
2. **"Reconstruir Índice (archivos en knowledge/)"** - Indexa todos los PDFs que ya están en la carpeta `knowledge/`

---

## OPCIÓN 1: Usar un PDF que ya está en knowledge/

### Paso 1: Verifica que el PDF existe
Tu PDF está en:
```
C:\Users\vasqu\OneDrive\Documentos\EAFIT\semestre5\proyecto-gemma\knowledge\ai-article.pdf
```

### Paso 2: Abre el Chatbot
```bash
python gemma3_270m_chat.py
```

### Paso 3: Presiona "Reconstruir Índice (archivos en knowledge/)"

![Botón en Gradio]
```
┌─────────────────────────────────────────────┐
│  [Subir e Indexar] [Subir archivo]         │
├─────────────────────────────────────────────┤
│  [Reconstruir Índice (archivos en knowledge/)]  ← PRESIONA ESTE
├─────────────────────────────────────────────┤
│  Status: ✓ Índice reconstruido exitosamente!
│
│          Archivos procesados: 1
└─────────────────────────────────────────────┘
```

### Paso 4: Espera a que termine
Verás un mensaje como:
```
✓ Índice reconstruido exitosamente!

Archivos procesados: 1
Total chunks indexados: (verifica con diagnose_rag.py)
```

### Paso 5: Prueba el chatbot
```
Usuario: ¿Qué es la inteligencia artificial?
```

---

## OPCIÓN 2: Usar la función de línea de comandos

Si prefieres indexar desde terminal sin abrir Gradio:

```bash
python build_index.py
```

Output:
```
======================================================================
CONSTRUIR ÍNDICE FAISS
======================================================================

1. Explorando directorio: knowledge
   ENCONTRADO: 1 archivos

2. Extrayendo texto de archivos...
   -> ai-article.pdf... OK (7 chunks, 17455 caracteres)

3. Generando embeddings...
4. Construyendo índice FAISS...
5. Guardando archivos...

======================================================================
ÍNDICE CONSTRUIDO EXITOSAMENTE!
======================================================================
```

---

## OPCIÓN 3: Agregar más PDFs y reconstruir

### Paso 1: Copia tus PDFs a la carpeta knowledge/
```
knowledge/
├── ai-article.pdf          (ya existe)
├── tu-otro-archivo.pdf     ← Copia aquí
└── otro-documento.txt      ← O archivos TXT/MD
```

### Paso 2: Presiona "Reconstruir Índice"
```
Status: ✓ Índice reconstruido exitosamente!

Archivos procesados: 3
```

---

## Verificar que funciona

Ejecuta el diagnóstico:
```bash
python diagnose_rag.py
```

Deberías ver:
```
[OK] PDF existe y contiene texto
[OK] Índice FAISS construido
[OK] Embeddings guardados
[OK] Metadata guardada

[OK] Sistema RAG esta completo. Si no responde correctamente:
  - Aumentar threshold de relevancia en chat_with_gemma()
  - Usar modelo de embeddings mas potente
  - Revisar chunking del texto
```

---

## Troubleshooting

### Error: "La carpeta 'knowledge/' no existe aún"
**Solución**: Sube un archivo primero con "Subir e Indexar"

### Error: "No hay archivos en la carpeta 'knowledge/'"
**Solución**: Asegúrate de tener un PDF en:
```
C:\Users\vasqu\OneDrive\Documentos\EAFIT\semestre5\proyecto-gemma\knowledge\ai-article.pdf
```

### El chatbot no responde correctamente
**Pasos**:
1. Ejecuta `diagnose_rag.py` para verificar el índice
2. Presiona "Reconstruir Índice" nuevamente
3. Cierra y abre el chatbot

### El botón "Reconstruir Índice" no aparece
**Solución**:
1. Asegúrate de que tienes la última versión de `gemma3_270m_chat.py`
2. Cierra Gradio completamente (`Ctrl + C`)
3. Ejecuta nuevamente: `python gemma3_270m_chat.py`

---

## Flujo Recomendado

```
1. Iniciar chatbot
   python gemma3_270m_chat.py

2. Presionar "Reconstruir Índice"
   ↓
   Esperar mensaje: "✓ Índice reconstruido"

3. Hacer una pregunta
   "¿Qué es la inteligencia artificial?"

4. Recibir respuesta basada en el PDF
```

---

## Resumen de Cambios

| Antes | Ahora |
|--------|-------|
| Solo "Subir e Indexar" (necesita archivo) | Dos opciones: "Subir e Indexar" + "Reconstruir Índice" |
| No podías indexar PDFs existentes | Indexa automáticamente PDFs en knowledge/ |
| Menos mensajes de estado | Mensajes claros y detallados |

---

## Archivos Involucrados

- `gemma3_270m_chat.py` - Interfaz con nuevo botón
- `knowledge/ai-article.pdf` - Tu PDF (7 chunks)
- `knowledge_index.faiss` - Índice generado
- `knowledge_embeddings.npy` - Vectores de embeddings
- `knowledge_metadata.json` - Metadatos de chunks

---

## Próximos Pasos

Una vez que funcione la indexación:

1. ✓ Presiona "Reconstruir Índice"
2. ✓ Prueba preguntas sobre el PDF
3. ✓ Agrega más PDFs si quieres
4. ✓ Ejecuta `diagnose_rag.py` para verificar estado

¡Listo! Tu sistema RAG debería funcionar correctamente ahora.
