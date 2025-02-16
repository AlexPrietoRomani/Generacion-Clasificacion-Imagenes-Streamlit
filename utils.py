import os
from diffusers import StableDiffusionPipeline
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torch

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

def detectar_rostro(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return None, "No se detectaron rostros en la imagen."

    rostro = face_locations[0]  # Tomamos el primer rostro detectado
    top, right, bottom, left = rostro
    pil_image = Image.fromarray(image[top:bottom, left:right])

    return pil_image, None

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
