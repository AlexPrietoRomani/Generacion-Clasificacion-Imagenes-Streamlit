import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from PIL import Image
import torch
import os
import time
from utils import classify_image, save_uploaded_file, ensure_model_downloaded,detectar_rostro, aplicar_estilo
from huggingface_hub import snapshot_download
import shutil
import io
# --------------------------------------------------------------------------------------------
# Configuración de la aplicación y manejo de modelos
# --------------------------------------------------------------------------------------------

model_id = "CompVis/stable-diffusion-v1-4"
local_model_dir = "./models/stable-diffusion-v1-4"

ensure_model_downloaded(model_id, local_model_dir)

if not os.path.exists(local_model_dir) or len(os.listdir(local_model_dir)) == 0:
    snapshot_download(repo_id=model_id, local_dir=local_model_dir)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

os.makedirs("temp", exist_ok=True)
generated_image_path = os.path.join("temp", "generated_image.png")

# --------------------------------------------------------------------------------------------
# Configuración inicial de Streamlit (interfaz)
# --------------------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Generador y Clasificador de Imágenes", page_icon="🎨")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
    }

    h1, h2, h3 {
        color: #333333;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }

    .stTextInput label, .stFileUploader label {
        font-weight: bold;
        color: #4a4a4a;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }

    .stButton button {
        background-color: #6C63FF;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.6em 1em;
        cursor: pointer;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }

    .stButton button:hover {
        background-color: #5a55d8;
    }

    .st-alert {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------------
# Información de PyTorch y CUDA para diagnóstico
# --------------------------------------------------------------------------------------------
gpu_available = torch.cuda.is_available()
torch_version = torch.__version__
cuda_version = torch.version.cuda
cudnn_version = torch.backends.cudnn.version()

st.sidebar.title("Opciones de Modelos y Diagnóstico GPU")

# Mostrar información de versiones en la barra lateral
st.sidebar.subheader("Información de PyTorch y CUDA")
st.sidebar.write(f"PyTorch version: {torch_version}")
st.sidebar.write(f"CUDA version según PyTorch: {cuda_version}")
if cudnn_version is not None:
    st.sidebar.write(f"cuDNN version: {cudnn_version}")
else:
    st.sidebar.write("cuDNN no detectado.")

# Botón para verificar GPU
if st.sidebar.button("Verificar GPU"):
    if gpu_available:
        st.sidebar.success("GPU detectada correctamente por PyTorch. Puedes seleccionar GPU abajo.")
    else:
        st.sidebar.error("No se detecta GPU. Posibles razones:\n"
                         "- PyTorch no se instaló con soporte CUDA.\n"
                         "- Faltan drivers NVIDIA/CUDA Toolkit.\n"
                         "- La GPU no es compatible con CUDA.\n\n"
                         "Verifica que has instalado PyTorch con CUDA. Por ejemplo:\n"
                         "`pip install torch --extra-index-url https://download.pytorch.org/whl/cu118`\n"
                         "También asegúrate de que tu GPU NVIDIA aparece en `nvidia-smi` y que los drivers estén al día.\n"
                         "Si ya hiciste esto, prueba reiniciar el entorno virtual y/o la máquina.")

# Seleccionar CPU o GPU si está disponible
if gpu_available:
    device_choice = st.sidebar.radio("Selecciona el dispositivo a usar:", ["CPU", "GPU"])
else:
    device_choice = st.sidebar.radio("Selecciona el dispositivo a usar:", ["CPU"])
    st.sidebar.warning("GPU no detectada. Usarás CPU.\n"
                       "Si tienes GPU y no aparece:\n"
                       "- Revisa `nvidia-smi`.\n"
                       "- Instala PyTorch con CUDA.\n"
                       "- Asegúrate de tener los drivers NVIDIA y CUDA Toolkit apropiados.\n"
                       "- Reinicia el entorno virtual y la máquina si es necesario.")

def get_device():
    if device_choice == "GPU" and not gpu_available:
        st.sidebar.warning("Se seleccionó GPU, pero no está disponible. Usando CPU.")
        return "cpu"
    elif device_choice == "GPU" and gpu_available:
        return "cuda"
    else:
        return "cpu"

# --------------------------------------------------------------------------------------------
# Función para cargar el modelo de generación (Stable Diffusion)
# --------------------------------------------------------------------------------------------
def load_generation_model():
    model_index_path = os.path.join(local_model_dir, "model_index.json")
    if (not os.path.exists(local_model_dir)
        or len(os.listdir(local_model_dir)) == 0
        or not os.path.exists(model_index_path)):
        st.write(f"El modelo {model_id} no está completamente descargado o falta 'model_index.json'. Descargando nuevamente...")
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)
        snapshot_download(repo_id=model_id, local_dir=local_model_dir)

    device = get_device()

    if device == "cuda":
        # Usar float16 en GPU
        pipe = StableDiffusionPipeline.from_pretrained(
            local_model_dir,
            torch_dtype=torch.float16
        )
    else:
        # Usar configuración por defecto (float32) en CPU
        pipe = StableDiffusionPipeline.from_pretrained(local_model_dir)

    pipe.to(device)
    return pipe

@st.cache_resource
def load_classification_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

with st.spinner("Cargando modelo de generación..."):
    generation_model = load_generation_model()
classification_model = load_classification_model()

st.markdown("""
Bienvenido a la aplicación de generación y clasificación de imágenes.

- Usa la barra lateral para diagnosticar si tu GPU está disponible.
- Si tu GPU no aparece, revisa la información y sugerencias proporcionadas.
- Genera imágenes a partir de un prompt (columna izquierda).
- Clasifica imágenes subidas o la última imagen generada (columna derecha).
""")

# Menú lateral para seleccionar la funcionalidad
opcion = st.sidebar.radio("Selecciona una funcionalidad:", 
                           ["Generación de Imágenes", "Clasificación de Imágenes", "Estilización de Rostro para CV"])

if opcion == "Generación de Imágenes":
    # === BLOQUE EXISTENTE: Generación de Imágenes ===
    st.header("Generación de Imágenes")
    st.markdown("Escribe un prompt y haz clic en **Generar Imagen** para crear una nueva imagen.")
    prompt = st.text_input("Prompt para la imagen:")
    
    if st.button("Generar Imagen"):
        if prompt:
            start_time = time.time()
            with st.spinner("Generando imagen..."):
                image = generation_model(prompt).images[0]
            elapsed = time.time() - start_time
            st.image(image, caption="Imagen Generada", use_container_width=True)
            image.save(generated_image_path)
            st.success(f"¡Imagen generada en {elapsed:.2f} segundos! La imagen está en 'temp'.")
        else:
            st.warning("Por favor, ingresa un prompt válido.")

elif opcion == "Clasificación de Imágenes":
    # === BLOQUE EXISTENTE: Clasificación de Imágenes ===
    st.header("Clasificación de Imágenes")
    st.markdown("Sube una imagen y haz clic en **Clasificar Imagen**.")
    uploaded_file = st.file_uploader("Subir imagen:", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image_path = save_uploaded_file(uploaded_file)
        image = Image.open(image_path)
        st.image(image, caption="Imagen Subida", use_container_width=True)

        if st.button("Clasificar Imagen"):
            start_time = time.time()
            with st.spinner("Clasificando imagen..."):
                results = classify_image(image_path, classification_model)
            elapsed = time.time() - start_time
            st.write("### Resultados de Clasificación:")
            for result in results:
                st.write(f"- **{result['label']}**: {result['score']:.2f}")
            st.success(f"Clasificación completada en {elapsed:.2f} segundos.")

    st.markdown("---")
    st.markdown("Si ya generaste una imagen, puedes clasificarla a continuación:")
    if st.button("Clasificar Imagen Generada"):
        if os.path.exists(generated_image_path):
            start_time = time.time()
            with st.spinner("Clasificando imagen generada..."):
                results = classify_image(generated_image_path, classification_model)
            elapsed = time.time() - start_time
            st.write("### Resultados de Clasificación de la Imagen Generada:")
            for result in results:
                st.write(f"- **{result['label']}**: {result['score']:.2f}")
            st.success(f"Clasificación completada en {elapsed:.2f} segundos.")
        else:
            st.warning("No se ha generado ninguna imagen aún. Por favor, genera una antes de clasificarla.")

elif opcion == "Estilización de Rostro para CV":
    # === NUEVO BLOQUE: Estilización de Rostro para CV ===
    st.header("Generador de Estilo de Rostro para CV")

    # Subida de imagen de rostro
    uploaded_face = st.file_uploader("Sube una imagen de tu rostro", type=["jpg", "png", "jpeg"])
    if uploaded_face is not None:
        st.image(uploaded_face, caption="Imagen original", use_column_width=True)

        # Detección de rostro
        rostro, error = detectar_rostro(uploaded_face)
        if error:
            st.error(error)
        else:
            st.image(rostro, caption="Rostro detectado", use_column_width=True)

            # Opciones de estilos disponibles
            estilos_disponibles = {
                "Van Gogh": "van_gogh.pth",
                "Picasso": "picasso.pth",
                "Sketch": "sketch.pth"
            }
            opcion_estilo = st.selectbox("Selecciona un estilo", list(estilos_disponibles.keys()))
            modelo_estilo_path = estilos_disponibles[opcion_estilo]

            # Aplicar transferencia de estilo
            with st.spinner("Aplicando estilo..."):
                estilo_image = aplicar_estilo(rostro, estilo_model_path=modelo_estilo_path)
            st.image(estilo_image, caption="Rostro estilizado", use_column_width=True)

            # Botón para descargar la imagen estilizada (usando BytesIO)
            buf = io.BytesIO()
            estilo_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Descargar imagen estilizada",
                data=byte_im,
                file_name="rostro_estilizado.png",
                mime="image/png"
            )