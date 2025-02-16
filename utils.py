import os
from diffusers import StableDiffusionPipeline
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torch
import numpy as np

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

def detectar_rostro(image_file):
    # Asegurarse de que el puntero esté al inicio
    image_file.seek(0)
    try:
        pil_image = Image.open(image_file)
    except Exception as e:
        return None, f"Error al abrir la imagen: {e}"
    
    # Convertir a RGB para forzar 3 canales (8-bit)
    pil_image = pil_image.convert("RGB")
    image = np.array(pil_image)
    
    # Si la imagen tuviera 4 canales (por ejemplo, RGBA), convertir nuevamente a RGB
    if image.ndim == 3 and image.shape[2] == 4:
        pil_image = Image.fromarray(image).convert("RGB")
        image = np.array(pil_image)
    
    # Asegurarse de que el tipo de dato sea uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return None, "No se detectaron rostros en la imagen."
    
    # Extraer el primer rostro detectado
    top, right, bottom, left = face_locations[0]
    rostro = image[top:bottom, left:right]
    return Image.fromarray(rostro), None

def aplicar_estilo(image, estilo_model_path="modelo_estilo.pth"):
    # Modelo preentrenado de transferencia de estilo
    estilo_modelo = torch.load(estilo_model_path)
    estilo_modelo.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output_tensor = estilo_modelo(input_tensor)

    output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
    return output_image

def aplicar_estilo_y_fondo(rostro, fondo_prompt, estilo_model_path="modelo_estilo.pth", background_size=(512, 512)):
    """
    Aplica transferencia de estilo al rostro y genera un fondo a partir del prompt dado.
    Combina ambos para obtener la imagen final.
    
    Parámetros:
      - rostro: Imagen PIL del rostro (ya extraída).
      - fondo_prompt: Texto con el prompt para generar el fondo.
      - estilo_model_path: Ruta al modelo de transferencia de estilo.
      - background_size: Tupla (ancho, alto) para la imagen de fondo.
    
    Retorna:
      - Imagen PIL (en modo RGBA) resultante.
    """
    # 1. Aplicar transferencia de estilo al rostro
    rostro_estilizado = aplicar_estilo(rostro, estilo_model_path)
    
    # 2. Generar el fondo usando Stable Diffusion
    model_id = "CompVis/stable-diffusion-v1-4"
    # Nota: para optimizar, deberías cargar el pipeline una sola vez
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    
    fondo = pipe(fondo_prompt, height=background_size[1], width=background_size[0]).images[0]
    
    # 3. Redimensionar el rostro para que ocupe, por ejemplo, el 30% del ancho del fondo
    fondo_width, fondo_height = fondo.size
    new_width = int(fondo_width * 0.3)
    aspect_ratio = rostro_estilizado.width / rostro_estilizado.height
    new_height = int(new_width / aspect_ratio)
    rostro_resized = rostro_estilizado.resize((new_width, new_height))
    
    # 4. Combinar el rostro sobre el fondo
    fondo_rgba = fondo.convert("RGBA")
    rostro_rgba = rostro_resized.convert("RGBA")
    position = ((fondo_width - new_width) // 2, (fondo_height - new_height) // 2)
    fondo_rgba.paste(rostro_rgba, position, mask=rostro_rgba)
    
    return fondo_rgba