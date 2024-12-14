import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from PIL import Image
import torch
import os
from utils import classify_image, save_uploaded_file,ensure_model_downloaded
from huggingface_hub import snapshot_download

# Verificar o descargar el modelo antes de usarlo
model_id = "CompVis/stable-diffusion-v1-4"
local_model_dir = "./models/stable-diffusion-v1-4"
ensure_model_downloaded(model_id, local_model_dir)

if not os.path.exists(local_model_dir) or len(os.listdir(local_model_dir)) == 0:
    snapshot_download(repo_id=model_id, local_dir=local_model_dir)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configuración inicial de Streamlit
st.set_page_config(layout="wide", page_title="Generador y Clasificador de Imágenes", page_icon="🎨")

# Estilos CSS personalizados para una apariencia más agradable
st.markdown("""
    <style>
    /* Contenedor Principal */
    .main {
        background-color: #f9f9f9;
        padding: 20px;
    }

    /* Títulos y textos */
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

# Cargar el modelo desde la ubicación local
def load_generation_model():
    model_index_path = os.path.join(local_model_dir, "model_index.json")
    
    # Si el directorio no existe o está vacío, o no existe el model_index.json, descargamos de nuevo
    if (not os.path.exists(local_model_dir) 
        or len(os.listdir(local_model_dir)) == 0 
        or not os.path.exists(model_index_path)):

        st.write(f"El modelo {model_id} no está completamente descargado o falta 'model_index.json'. Descargando nuevamente...")

        # Si el directorio existe pero no tiene el archivo necesario, lo borramos para evitar conflictos
        if os.path.exists(local_model_dir):
            # Eliminar todo el directorio del modelo incompleto
            import shutil
            shutil.rmtree(local_model_dir)

        # Volver a descargar el modelo con snapshot_download o directamente con from_pretrained
        snapshot_download(repo_id=model_id, local_dir=local_model_dir)
        
        # Ahora cargamos desde el directorio ya descargado
        pipe = StableDiffusionPipeline.from_pretrained(
            local_model_dir, 
            torch_dtype=torch.float16
        )
        
    else:
        st.write(f"Modelo ya disponible en {local_model_dir}")
        pipe = StableDiffusionPipeline.from_pretrained(
            local_model_dir, 
            torch_dtype=torch.float16
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe



# Cargar el modelo de clasificación
@st.cache_resource
def load_classification_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

st.sidebar.title("Opciones de Modelos")
st.sidebar.markdown("Selecciona las acciones que desees realizar en la aplicación.")

# Modelos
with st.spinner("Cargando modelo de generación..."):
    generation_model = load_generation_model()
classification_model = load_classification_model()

# Descripción general
st.markdown("""
Bienvenido a la aplicación de generación y clasificación de imágenes.
- En la **sección izquierda** tienes las opciones de modelos.
- En la **sección derecha** podrás subir una imagen para clasificar.
- En la **sección de la izquierda** también puedes generar una imagen a partir de un texto (prompt).

¡Disfruta la experiencia!
""")

# Layout de la aplicación
col1, col2 = st.columns(2)

# Sección de Generación de Imágenes (col1)
with col1:
    st.header("Generación de Imágenes")
    st.markdown("Introduce un texto descriptivo (prompt) y haz clic en **Generar Imagen** para crear una nueva imagen.")
    prompt = st.text_input("Prompt para la imagen:")
    
    if st.button("Generar Imagen"):
        if prompt:
            with st.spinner("Generando imagen..."):
                image = generation_model(prompt).images[0]
                st.image(image, caption="Imagen Generada", use_column_width=True)
                image.save("generated_image.png")
                st.success("¡Imagen generada con éxito! Encuéntrala debajo o en la carpeta local.")
        else:
            st.warning("Por favor, ingresa un prompt válido.")

    st.markdown("---")

# Sección de Clasificación de Imágenes (col2)
with col2:
    st.header("Clasificación de Imágenes")
    st.markdown("Sube una imagen desde tu computadora y haz clic en **Clasificar Imagen** para conocer las etiquetas más probables.")
    uploaded_file = st.file_uploader("Subir imagen:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        image_path = save_uploaded_file(uploaded_file)
        image = Image.open(image_path)
        st.image(image, caption="Imagen Subida", use_column_width=True)
        
        if st.button("Clasificar Imagen"):
            with st.spinner("Clasificando imagen..."):
                results = classify_image(image_path, classification_model)
                st.write("### Resultados de Clasificación:")
                for result in results:
                    st.write(f"- **{result['label']}**: {result['score']:.2f}")
    
    st.markdown("---")
    st.markdown("Si ya has generado una imagen, también puedes clasificarla a continuación.")
    if st.button("Clasificar Imagen Generada"):
        if os.path.exists("generated_image.png"):
            with st.spinner("Clasificando imagen generada..."):
                results = classify_image("generated_image.png", classification_model)
                st.write("### Resultados de Clasificación de la Imagen Generada:")
                for result in results:
                    st.write(f"- **{result['label']}**: {result['score']:.2f}")
        else:
            st.warning("No se ha generado ninguna imagen aún. Por favor, genera una antes de clasificarla.")

