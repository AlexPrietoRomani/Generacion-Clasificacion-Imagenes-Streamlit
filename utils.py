import os
from diffusers import StableDiffusionPipeline
from PIL import Image

def save_uploaded_file(uploaded_file):
    """Guarda el archivo subido temporalmente."""
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def classify_image(image_path, model):
    """Clasifica una imagen usando el modelo de clasificación."""
    image = Image.open(image_path).convert("RGB")
    results = model(image)
    return results

def ensure_model_downloaded(model_id, local_dir):
    """Verifica si el modelo ya está descargado, de lo contrario, lo descarga."""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        print(f"Descargando el modelo {model_id} en {local_dir}...")
        StableDiffusionPipeline.from_pretrained(model_id, cache_dir=local_dir)
    else:
        print(f"Modelo ya está disponible en {local_dir}")