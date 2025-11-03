"""
Chat Local con Gemma 3 270M Instruct - VERSIÓN CORREGIDA ✅
Proyecto: Especialización SLM Gemma
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
MODEL_NAME = "google/gemma-3-270m-it"  # ✅ VERSIÓN INSTRUCTION-TUNED (la correcta!)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path donde el usuario puede poner archivos de conocimiento (txt, md, jsonl, etc.)
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "knowledge_index.faiss")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "knowledge_embeddings.npy")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
METADATA_PATH = os.path.join(os.path.dirname(__file__), "knowledge_metadata.json")

print("="*60)
print("🚀 CHAT LOCAL CON GEMMA 3 270M INSTRUCT")
print("   Proyecto SLM - Universidad EAFIT 2025")
print("="*60)
print(f"📱 Dispositivo: {DEVICE.upper()}")
print(f"🔧 Modelo: {MODEL_NAME}")
print("✅ Usando versión INSTRUCTION-TUNED (sigue instrucciones)")
print()

# ====== CARGA DEL MODELO ======
print(f"⏳ Cargando Gemma 3 270M Instruct...")
print("   (Primera vez descargará ~241MB)")
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
        print(f"⏳ Cargando adaptador LoRA desde {LORA_DIR}...")
        model = PeftModel.from_pretrained(model, LORA_DIR, device_map="auto" if DEVICE == "cuda" else None)
        print("✅ Adaptador LoRA cargado correctamente.")
    except Exception as e:
        print("⚠️ No se pudo cargar el adaptador LoRA:", e)
        print("Continuando con el modelo base...")

print("✅ ¡Gemma 3 270M Instruct cargado exitosamente!")
print("✅ Este modelo SÍ sigue instrucciones correctamente")
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


# ====== FUNCIÓN DE CHAT ======
def chat_with_gemma(message, history, temperature, max_tokens):
    """
    Función de chat con Gemma 3 270M Instruct
    Usa el formato correcto para el modelo instruction-tuned
    """
    
    # Recuperación de contexto local (RAG) usando FAISS + metadata (chunks)
    context_text = ""
    if SentenceTransformer is not None and faiss is not None and np is not None:
        try:
            if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
                embed_model = SentenceTransformer(EMBED_MODEL_NAME)
                index = faiss.read_index(FAISS_INDEX_PATH)
                q_emb = embed_model.encode([message])
                D, I = index.search(np.array(q_emb).astype('float32'), 3)
                # cargar metadata (lista de fragments)
                with open(METADATA_PATH, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                for idx in I[0]:
                    if idx < len(metadata):
                        entry = metadata[idx]
                        context_text += f"\n---\nSource: {entry.get('source','unknown')}\n{entry.get('text','')}\n"
        except Exception:
            context_text = ""

    # Construir prompt en formato de chat simple
    # Gemma 3 270M-it responde mejor con formato directo
    if len(history) > 0:
        # Incluir últimos 3 intercambios para contexto
        conversation = ""
        for user_msg, bot_msg in history[-3:]:
            conversation += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        # añadir contexto recuperado antes de la última consulta
        if context_text:
            conversation += f"Context:{context_text}\n"
        conversation += f"User: {message}\nAssistant:"
    else:
        conversation = ""
        if context_text:
            conversation += f"Context:{context_text}\n"
        conversation += f"User: {message}\nAssistant:"
    
    # Tokenizar
    inputs = tokenizer(
        conversation, 
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(DEVICE)
    
    # Generar respuesta
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0.1,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decodificar
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraer solo la respuesta del asistente
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response[len(conversation):].strip()
    
    # Limpiar respuesta (quitar posibles "User:" que aparezcan)
    if "User:" in response:
        response = response.split("User:")[0].strip()
    
    return response

# ====== INTERFAZ GRADIO ======
with gr.Blocks(
    title="Chat Gemma 3 270M Instruct 💬",
    theme=gr.themes.Soft(primary_hue="blue")
) as demo:
    
    gr.Markdown(
        """
        # 🤖 Chat Local con Gemma 3 270M Instruct
        ### Proyecto de Especialización SLM - Universidad EAFIT 2025
        
        ✅ **Versión CORREGIDA** - Usando modelo **instruction-tuned** que SÍ sigue instrucciones correctamente.
        
        **Modelo**: `google/gemma-3-270m-it` (instruction-tuned)
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=("👤", "🤖"),
                bubble_full_width=False
            )
            # Área para subir archivos al índice de conocimiento
            with gr.Row():
                file_input = gr.File(label="Subir archivo (txt, md, pdf)")
                upload_btn = gr.Button("📥 Subir e indexar")
            index_status = gr.Markdown(visible=True)
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Escribe tu mensaje aquí...",
                    show_label=False,
                    scale=9,
                    container=False
                )
                send_btn = gr.Button("Enviar 📤", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Limpiar Chat", scale=1)
                retry_btn = gr.Button("🔄 Reintentar", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuración")
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="🌡️ Temperatura",
                info="Creatividad de las respuestas"
            )
            
            max_tokens = gr.Slider(
                minimum=50,
                maximum=300,
                value=150,
                step=25,
                label="📏 Tokens Máximos",
                info="Longitud de la respuesta"
            )
            
            gr.Markdown(
                f"""
                ---
                ### 📊 Info del Sistema
                - **Modelo**: Gemma 3 270M **Instruct** ✅
                - **Tipo**: Instruction-Tuned
                - **Dispositivo**: {DEVICE.upper()}
                - **Estado**: 🟢 Activo
                
                ---
                ### 💡 Diferencia con Pre-trained
                
                **Instruction-Tuned (IT)** ✅:
                - Sigue instrucciones
                - Responde preguntas coherentemente
                - Formato de chat
                
                **Pre-trained (PT)** ❌:
                - Solo continúa texto
                - No sigue instrucciones
                - Genera texto random
                """
            )
    
    # Ejemplos mejorados
    gr.Examples(
        examples=[
            "Hola, ¿cómo estás?",
            "¿Qué es la inteligencia artificial?",
            "Explícame qué es un transformer",
            "Dame 3 ventajas de los modelos pequeños",
            "¿Cómo funciona el fine-tuning?"
        ],
        inputs=msg,
        label="💬 Preguntas de ejemplo"
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
            return "No se recibió archivo."
        saved = save_uploaded_file(uploaded_file)
        if not saved:
            return "No se pudo guardar el archivo subido."
        try:
            build_faiss_index(KNOWLEDGE_DIR)
            return f"Archivo guardado en `{saved}` y índice construido/actualizado."
        except Exception as e:
            return f"Error construyendo índice: {e}"
    
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
    
    clear_btn.click(lambda: [], outputs=chatbot)
    
    retry_btn.click(
        retry_last,
        inputs=[chatbot, temperature, max_tokens],
        outputs=chatbot
    )

# ====== LANZAR APLICACIÓN ======
if __name__ == "__main__":
    print("="*60)
    print("🌐 Abriendo interfaz web en http://127.0.0.1:7860")
    print("="*60)
    print()
    print("💡 NOTA IMPORTANTE:")
    print("   Ahora usando modelo INSTRUCTION-TUNED")
    print("   Las respuestas serán coherentes ✅")
    print()
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )