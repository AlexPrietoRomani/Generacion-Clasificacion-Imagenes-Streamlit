from diffusers import StableDiffusionPipeline
import torch  # Asegúrate de importar torch

# Define el modelo y la ruta de destino
model_id = "CompVis/stable-diffusion-v1-4"

# Descargar y guardar el modelo localmente
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.save_pretrained("./models/stable-diffusion-v1-4")

print("Modelo descargado y guardado en './models/stable-diffusion-v1-4'")
