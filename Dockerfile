# Imagen base de Python con CUDA (para GPU). Cambiar a una imagen base más ligera si usas solo CPU.
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requisitos para instalar las dependencias
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Crear un directorio para modelos y descargar el modelo desde HuggingFace
RUN mkdir -p /models/stable-diffusion-v1-4

# Copia el archivo .whl de dlib (asegúrate de que sea la versión manylinux para Linux)
COPY dlib-19.24.1-cp311-cp311-win_amd64.whl /tmp/

# Descargar el modelo de HuggingFace
RUN python3 -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', cache_dir='/models/stable-diffusion-v1-4')"

# Copiar el resto de los archivos de la aplicación al contenedor
COPY . .

# Comando para iniciar la aplicación Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]