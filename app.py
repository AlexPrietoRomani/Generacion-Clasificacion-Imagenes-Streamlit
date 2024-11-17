import streamlit as st
from utils import ensure_model_downloaded
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from PIL import Image
import torch
import os
from utils import classify_image, save_uploaded_file
import os

# Verificar o descargar el modelo antes de usarlo
model_id = "CompVis/stable-diffusion-v1-4"
local_model_dir = "./models/stable-diffusion-v1-4"
ensure_model_downloaded(model_id, local_model_dir)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configuración inicial de Streamlit
st.set_page_config(layout="wide")

# Cargar el modelo desde la ubicación local
def load_generation_model():
    pipe = StableDiffusionPipeline.from_pretrained(local_model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe

# Cargar el modelo de clasificación
@st.cache_resource
def load_classification_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

# Modelos
st.sidebar.title("Opciones de Modelos")
generation_model = load_generation_model()
classification_model = load_classification_model()

# app.py
with st.spinner("Cargando modelo de generación..."):
    generation_model = load_generation_model()

# Layout de la aplicación
col1, col2 = st.columns(2)

# Sección de Generación de Imágenes (col1)
with col1:
    st.header("Generación de Imágenes")
    prompt = st.text_input("Escribe tu solicitud para generar una imagen:")
    
    if st.button("Generar Imagen"):
        if prompt:
            with st.spinner("Generando imagen..."):
                image = generation_model(prompt).images[0]
                st.image(image, caption="Imagen Generada", use_column_width=True)
                image.save("generated_image.png")
                st.success("¡Imagen generada con éxito!")
        else:
            st.warning("Por favor, ingresa un prompt válido.")

# Sección de Clasificación de Imágenes (col2)
with col2:
    st.header("Clasificación de Imágenes")
    uploaded_file = st.file_uploader("Sube una imagen para clasificar:")
    
    if uploaded_file:
        image_path = save_uploaded_file(uploaded_file)
        image = Image.open(image_path)
        st.image(image, caption="Imagen Subida", use_column_width=True)
        
        if st.button("Clasificar Imagen"):
            with st.spinner("Clasificando imagen..."):
                results = classify_image(image_path, classification_model)
                st.write("Resultados de Clasificación:")
                for result in results:
                    st.write(f"- **{result['label']}**: {result['score']:.2f}")
    
    # Clasificar imagen generada
    if st.button("Clasificar Imagen Generada"):
        if os.path.exists("generated_image.png"):
            with st.spinner("Clasificando imagen generada..."):
                results = classify_image("generated_image.png", classification_model)
                st.write("Resultados de Clasificación de la Imagen Generada:")
                for result in results:
                    st.write(f"- **{result['label']}**: {result['score']:.2f}")
        else:
            st.warning("No se ha generado ninguna imagen aún.")

