"""
Gemma 3 270M Chat with RAG
Local chatbot con búsqueda en documentos
Autores: Felipe Castro Jaimes, Nicolás Vázquez, José Jiménez
Universidad EAFIT, 2025
"""

import os
import pathlib
import json
import uuid
from typing import List
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

# Configurar para usar cache local - desactivamos intentos de descarga en línea
# Esto previene que se intente descargar modelos si no hay internet
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.expanduser('~/.cache/huggingface/hub')

# Optional libraries for RAG and LoRA
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except Exception:
    SentenceTransformer = None
    faiss = None
    np = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ====== CONFIGURACIÓN - GEMMA 3 270M INSTRUCT ======
MODEL_NAME = "google/gemma-3-270m-it"  # Instruction-tuned version
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path donde el usuario puede poner archivos de conocimiento (txt, md, jsonl, etc.)
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "knowledge_index.faiss")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "knowledge_embeddings.npy")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
METADATA_PATH = os.path.join(os.path.dirname(__file__), "knowledge_metadata.json")

print("="*60)
print("Gemma 3 270M Chat")
print("="*60)
print(f"Dispositivo: {DEVICE.upper()}")
print(f"Modelo: {MODEL_NAME}")
print()

# Cargando modelo
print("Cargando modelo...")
print()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Cargar modelo optimizado
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    low_cpu_mem_usage=True
)

if DEVICE == "cpu":
    model = model.to(DEVICE)

# Si existe un adaptador LoRA (entrenado previamente) lo cargamos aquí
LORA_DIR = os.path.join(os.path.dirname(__file__), "lora_adapter")
if PeftModel is not None and os.path.isdir(LORA_DIR):
    try:
        print(f"Cargando adaptador LoRA desde {LORA_DIR}...")
        model = PeftModel.from_pretrained(model, LORA_DIR, device_map="auto" if DEVICE == "cuda" else None)
        print("LoRA adapter cargado.")
    except Exception as e:
        print("LoRA adapter no encontrado, continuando...")

print("Modelo cargado exitosamente")
print()


#######################
# Helpers: extracción, chunking y creación de índice FAISS
#######################

def extract_text_from_file(path: str) -> str:
    """Extrae texto de pdf, txt o md. Devuelve cadena vacía si no puede."""
    p = pathlib.Path(path)
    if not p.exists():
        return ""
    try:
        if p.suffix.lower() == ".pdf":
            if PdfReader is None:
                return ""
            text = []
            try:
                reader = PdfReader(str(p))
                for page in reader.pages:
                    try:
                        text.append(page.extract_text() or "")
                    except Exception:
                        continue
            except Exception:
                return ""
            return "\n".join(text)
        else:
            # txt, md, other text files
            try:
                return p.read_text(encoding='utf-8')
            except Exception:
                try:
                    return p.read_text(encoding='latin-1')
                except Exception:
                    return ""
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Divide el texto en chunks aproximados por palabras."""
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def build_faiss_index(knowledge_dir: str, chunk_size: int = 500, overlap: int = 50):
    """Construye/reescribe el índice FAISS y la metadata desde los archivos en knowledge_dir."""
    if SentenceTransformer is None or faiss is None or np is None:
        raise RuntimeError("Dependencias de RAG no están instaladas (sentence-transformers/faiss/numpy)")

    docs = []
    metadata = []
    # recorrer archivos y fragmentar
    p = pathlib.Path(knowledge_dir)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

    for fp in sorted(p.glob("**/*")):
        if fp.is_file():
            text = extract_text_from_file(str(fp))
            if not text:
                continue
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for c in chunks:
                metadata.append({
                    "id": len(metadata),
                    "source": str(fp.name),
                    "text": c[:2000]
                })
                docs.append(c)

    if not docs:
        # no hay docs para indexar
        return

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embs = embed_model.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    embs = embs.astype('float32')

    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    # guardar index y metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embs)
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_uploaded_file(uploaded) -> str:
    """Guarda el archivo subido en la carpeta knowledge y devuelve la ruta guardada o '' si falla."""
    if uploaded is None:
        return ""
    try:
        # uploaded puede ser una ruta string, un objeto con attribute 'name', o un dict (gradio)
        src_path = None
        if isinstance(uploaded, str):
            src_path = uploaded
        elif isinstance(uploaded, dict):
            # gradio a veces pasa {'name': ..., 'tmp_path': ...}
            src_path = uploaded.get('name') or uploaded.get('tmp_path') or uploaded.get('file')
        else:
            src_path = getattr(uploaded, 'name', None)

        if not src_path:
            return ""

        dest_dir = pathlib.Path(KNOWLEDGE_DIR)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{uuid.uuid4().hex}_{pathlib.Path(src_path).name}"
        import shutil
        shutil.copy(src_path, str(dest))
        return str(dest)
    except Exception:
        return ""


# ====== FUNCIÓN DE CHAT MEJORADA ======
def chat_with_gemma(message, history, temperature, max_tokens):
    """
    Chat mejorado con Gemma 3 270M Instruct
    - System prompt para mejores respuestas
    - RAG optimizado
    - Parámetros de generación ajustados
    """

    # System prompt SIMPLE para Gemma 3
    SYSTEM_PROMPT = """Eres un asistente inteligente.
Responde siempre de manera clara y directa.
Usa la información disponible para responder."""

    # Recuperación RAG mejorada - Con soporte offline
    context_text = ""
    if SentenceTransformer is not None and faiss is not None and np is not None:
        try:
            if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
                # Cargar embeddings desde cache LOCAL sin intentar descargar
                # Esto permite funcionalidad offline completa
                try:
                    # Usar cache_folder para apuntar al directorio correcto
                    # Forzar que no intente descargar con local_files_only=True
                    embed_model = SentenceTransformer(
                        EMBED_MODEL_NAME,
                        cache_folder=os.path.expanduser('~/.cache/huggingface/hub'),
                        local_files_only=True  # CRÍTICO: Solo usar archivos locales
                    )

                    index = faiss.read_index(FAISS_INDEX_PATH)
                    q_emb = embed_model.encode([message])
                    # Buscar top-2 documentos
                    D, I = index.search(np.array(q_emb).astype('float32'), 2)
                    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    # Incluir TODOS los resultados relevantes (muy permisivo)
                    for dist, idx in zip(D[0], I[0]):
                        if idx < len(metadata):
                            entry = metadata[idx]
                            text = entry.get('text', '')
                            # Limitar tamaño de cada chunk para evitar que sea muy largo
                            text = text[:300]  # Máximo 300 chars por chunk
                            context_text += f"{text}\n"
                except Exception as e:
                    # Si falla (ej: modelo no descargado aún), continuar sin RAG
                    print(f"Aviso: RAG no disponible - {str(e)[:100]}")
                    print("       (Ejecuta: python descargar_modelos.py para descargar)")
                    context_text = ""
        except Exception as e:
            # Fallback por si hay error en cualquier parte del RAG
            context_text = ""

    # Construir prompt SIMPLE y CORTO
    prompt = f"{SYSTEM_PROMPT}\n\n"

    # Agregar SOLO el contexto RAG más relevante (sin explicaciones extra)
    if context_text:
        prompt += f"Información:\n{context_text}\n"

    # NO agregar historial para mantener prompt corto
    # (Historial cansa al modelo pequeño)

    # Agregar mensaje actual
    prompt += f"Pregunta: {message}\n\nRespuesta:"

    # Tokenizar con truncation cuidadosa
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(DEVICE)

    # Parámetros de generación AGRESIVOS para forzar respuestas
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, 300),  # Respuestas más largas
            temperature=min(temperature, 0.6),    # Balance entre creatividad y coherencia
            do_sample=True,
            top_p=0.88,                           # Moderado
            top_k=40,                             # Moderado
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,               # FUERTE para evitar repeticiones
            no_repeat_ngram_size=3                # Evitar 3-gramas repetidos
        )

    # Decodificar respuesta
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer la respuesta DESPUÉS del prompt
    # El modelo responde después de "Respuesta:"
    if "Respuesta:" in full_response:
        response = full_response.split("Respuesta:")[-1].strip()
    else:
        # Fallback: tomar todo después del prompt original
        response = full_response[len(prompt):].strip()

    # Limpiar respuestas truncadas o mal formadas
    # Quitar cualquier cosa que pare en mitad de oración
    if "Pregunta:" in response:
        response = response.split("Pregunta:")[0].strip()
    if "Información:" in response:
        response = response.split("Información:")[0].strip()

    # Limpieza básica
    response = response.strip()

    # Detectar y limpiar repeticiones excesivas
    lineas = response.split('\n')
    lineas_limpias = []
    ultima_linea = ""

    for linea in lineas:
        linea = linea.strip()
        # Si la línea es muy similar a la anterior (>80% coincidencia), skip
        if linea and ultima_linea:
            coincidencias = sum(1 for a, b in zip(linea, ultima_linea) if a == b)
            similitud = coincidencias / len(ultima_linea)
            if similitud < 0.8:  # Si no es casi idéntica, agregar
                lineas_limpias.append(linea)
                ultima_linea = linea
        elif linea:
            lineas_limpias.append(linea)
            ultima_linea = linea

    response = " ".join(lineas_limpias)

    # Si la respuesta es vacía o muy corta, intentar recuperarla del contexto
    if len(response) < 5:
        # Si tenemos contexto pero no generó respuesta, tomar del contexto
        if context_text:
            # Extraer primera oración del contexto como respuesta
            primera_linea = context_text.split('\n')[0][:250]
            response = f"Basándome en la información disponible: {primera_linea}"
        else:
            response = "No tengo información disponible para responder. Por favor, carga un PDF."

    # Limitar a 800 caracteres máximo (más permisivo)
    if len(response) > 800:
        # Cortar en la última oración completa
        response = response[:800]
        # Encontrar último punto, signo de exclamación o interrogación
        for char in '.!?':
            if char in response:
                response = response.rsplit(char, 1)[0] + char
                break

    return response

# ====== INTERFAZ GRADIO ======
with gr.Blocks(
    title="Gemma Chat",
    theme=gr.themes.Default()
) as demo:

    gr.Markdown(
        """
        # Gemma 3 270M Chat

        Chatbot local con búsqueda en documentos (RAG)
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                bubble_full_width=False
            )

            with gr.Row():
                file_input = gr.File(label="Archivo (txt, md, pdf)")
                upload_btn = gr.Button("Subir e Indexar", scale=1)

            with gr.Row():
                rebuild_btn = gr.Button("Reconstruir Índice (archivos en knowledge/)", scale=2, variant="secondary")

            index_status = gr.Markdown(visible=True)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Escribe tu pregunta...",
                    show_label=False,
                    scale=9,
                    container=False
                )
                send_btn = gr.Button("Enviar", scale=1, variant="primary")

            with gr.Row():
                clear_btn = gr.Button("Limpiar", scale=1)
                retry_btn = gr.Button("Reintentar", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### Configuración")

            temperature = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.4,
                step=0.1,
                label="Temperatura",
                info="0.1=consistente, 0.9=creativo"
            )

            max_tokens = gr.Slider(
                minimum=50,
                maximum=200,
                value=120,
                step=25,
                label="Tokens",
                info="Longitud máxima (recomendado 100-150)"
            )

            gr.Markdown(
                f"""
                ### Sistema

                **Modelo**: Gemma 3 270M
                **Dispositivo**: {DEVICE.upper()}
                **RAG**: Habilitado

                ---

                **Consejos:**
                - Sube PDFs para búsqueda
                - Ajusta temperatura para más/menos creatividad
                - Presiona Enter o click en Enviar
                """
            )
    
    # Ejemplos
    gr.Examples(
        examples=[
            "Hola, ¿cómo estás?",
            "¿Qué es la inteligencia artificial?",
            "Explícame qué es un transformer",
            "Ventajas de los modelos pequeños",
            "¿Cómo funciona el fine-tuning?"
        ],
        inputs=msg,
        label="Ejemplos"
    )
    
    # ====== EVENTOS ======
    def respond(message, chat_history, temp, tokens):
        if not message.strip():
            return chat_history, ""
        
        bot_response = chat_with_gemma(message, chat_history, temp, tokens)
        chat_history.append((message, bot_response))
        return chat_history, ""


    def upload_and_index(uploaded_file):
        """Callback para guardar y construir el índice FAISS con el archivo subido."""
        if not uploaded_file:
            return "No se recibió archivo. Presiona 'Reconstruir Índice' para indexar archivos existentes en knowledge/"
        saved = save_uploaded_file(uploaded_file)
        if not saved:
            return "No se pudo guardar el archivo subido."
        try:
            build_faiss_index(KNOWLEDGE_DIR)
            return f"✓ Archivo guardado e índice actualizado correctamente.\n\nArchivos indexados: {len(pathlib.Path(KNOWLEDGE_DIR).glob('**/*'))}"
        except Exception as e:
            return f"Error construyendo índice: {e}"

    def rebuild_index():
        """Callback para reconstruir el índice con los archivos existentes en knowledge/"""
        try:
            knowledge_path = pathlib.Path(KNOWLEDGE_DIR)
            if not knowledge_path.exists():
                return "La carpeta 'knowledge/' no existe aún. Sube un archivo primero."

            files = list(knowledge_path.glob("**/*"))
            files = [f for f in files if f.is_file()]

            if not files:
                return "No hay archivos en la carpeta 'knowledge/'. Sube un PDF, TXT o MD."

            build_faiss_index(KNOWLEDGE_DIR)
            num_files = len(files)
            return f"✓ Índice reconstruido exitosamente!\n\nArchivos procesados: {num_files}\nTotal chunks indexados: (verifica con diagnose_rag.py)"
        except Exception as e:
            return f"Error reconstruyendo índice: {str(e)}"
    
    def retry_last(chat_history, temp, tokens):
        if not chat_history:
            return chat_history
        
        last_message = chat_history[-1][0]
        chat_history = chat_history[:-1]
        
        bot_response = chat_with_gemma(last_message, chat_history, temp, tokens)
        chat_history.append((last_message, bot_response))
        return chat_history
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, msg]
    )
    
    send_btn.click(
        respond,
        inputs=[msg, chatbot, temperature, max_tokens],
        outputs=[chatbot, msg]
    )
    upload_btn.click(
        upload_and_index,
        inputs=[file_input],
        outputs=[index_status]
    )

    rebuild_btn.click(
        rebuild_index,
        inputs=[],
        outputs=[index_status]
    )

    clear_btn.click(lambda: [], outputs=chatbot)
    
    retry_btn.click(
        retry_last,
        inputs=[chatbot, temperature, max_tokens],
        outputs=chatbot
    )

# Lanzar interfaz
if __name__ == "__main__":
    print("="*60)
    print("Iniciando servidor web")
    print("URL: http://127.0.0.1:7860")
    print("="*60)
    print()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )