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

    # System prompt para guiar mejor el modelo
    SYSTEM_PROMPT = """Eres un asistente útil y preciso.
Responde de manera clara, concisa y coherente.
Si no conoces la respuesta, di "No tengo información sobre eso" en lugar de inventar.
Evita respuestas confusas o incoherentes."""

    # Recuperación RAG mejorada
    context_text = ""
    if SentenceTransformer is not None and faiss is not None and np is not None:
        try:
            if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
                embed_model = SentenceTransformer(EMBED_MODEL_NAME)
                index = faiss.read_index(FAISS_INDEX_PATH)
                q_emb = embed_model.encode([message])
                # Buscar top-3 documentos (menos conservador)
                D, I = index.search(np.array(q_emb).astype('float32'), 3)
                with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                # Incluir resultados con distancia más permisiva
                for dist, idx in zip(D[0], I[0]):
                    if idx < len(metadata) and dist < 3.0:  # Threshold aumentado a 3.0
                        entry = metadata[idx]
                        source = entry.get('source', 'unknown')
                        text = entry.get('text', '')
                        context_text += f"Fuente: {source}\n{text}\n\n"
        except Exception:
            context_text = ""

    # Construir prompt mejorado
    prompt = f"{SYSTEM_PROMPT}\n\n"

    # Agregar contexto RAG si está disponible
    if context_text:
        prompt += f"INFORMACIÓN DE REFERENCIA RELEVANTE:\n{context_text}\n"
        prompt += "Usa la información anterior para responder la pregunta del usuario.\n\n"
    else:
        prompt += "No hay documentos disponibles para consultar.\n\n"

    # Agregar historial limitado
    if len(history) > 0:
        prompt += "CONVERSACIÓN ANTERIOR:\n"
        for user_msg, bot_msg in history[-2:]:  # Solo últimos 2 intercambios
            prompt += f"Usuario: {user_msg}\n"
            prompt += f"Asistente: {bot_msg}\n"
        prompt += "\n"

    # Agregar mensaje actual
    prompt += f"Usuario: {message}\nAsistente:"

    # Tokenizar con truncation cuidadosa
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    ).to(DEVICE)

    # Parámetros de generación optimizados para coherencia
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(max_tokens, 250),  # Permitir respuestas más largas
            temperature=min(temperature, 0.5),    # Reducir para mayor consistencia
            do_sample=True,
            top_p=0.85,                           # Slightly más restrictivo
            top_k=30,                             # Más restrictivo que antes
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,               # Ligero penalty
            no_repeat_ngram_size=2,               # No repetir bigramas
            length_penalty=1.0                    # Neutral en longitud
        )

    # Decodificar respuesta
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo la parte del asistente
    if "Asistente:" in full_response:
        response = full_response.split("Asistente:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()

    # Limpiar (quitar "Usuario:" o referencias a sistema si aparecen)
    if "Usuario:" in response:
        response = response.split("Usuario:")[0].strip()

    # Quitar prompts residuales
    for phrase in ["INFORMACIÓN DE REFERENCIA", "CONVERSACIÓN ANTERIOR", "Asistente:", "[INST]", "[/INST]"]:
        if phrase in response:
            response = response.split(phrase)[0].strip()

    # Limpieza de espacios múltiples
    response = " ".join(response.split())

    # Validación: respuesta debe ser significativa
    if not response or len(response.strip()) < 5:
        if context_text:
            response = "Lo siento, no pude generar una respuesta basada en los documentos disponibles. Intenta con una pregunta más específica."
        else:
            response = "No tengo documentos cargados para responder. Por favor, carga un PDF usando el botón 'Indexar'."
    elif any(phrase in response.lower() for phrase in ["no tengo información", "no sé", "no disponible"]):
        # Detectar cuando el modelo dice que no sabe
        if context_text:
            response = "Disculpa, aunque tengo información relacionada, no puedo formular una respuesta clara. Intenta reformular tu pregunta."

    # Limitar a 600 caracteres para respuestas más completas
    if len(response) > 600:
        response = response[:600].rsplit(" ", 1)[0] + "..."

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