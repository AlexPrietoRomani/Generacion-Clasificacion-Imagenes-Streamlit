# Practica calificada 2: Generación y Clasificación de Imágenes con Streamlit

Este proyecto es una aplicación web interactiva creada con Streamlit que permite a los usuarios:

1. Generar imágenes a partir de descripciones de texto utilizando el modelo preentrenado Stable Diffusion.
2. Clasificar imágenes (subidas o generadas) utilizando el modelo preentrenado Vision Transformer (ViT) de HuggingFace.

## Características
- Generación de Imágenes (Columna 1):

    - Permite a los usuarios ingresar un texto descriptivo (prompt).
    - Genera una imagen realista basada en el texto proporcionado.
    - Guarda la imagen generada localmente.

- Clasificación de Imágenes (Columna 2):

    - Clasifica imágenes subidas por el usuario en categorías específicas.
    - Permite clasificar la imagen generada en la columna 1.
    - Devuelve etiquetas con puntajes de confianza.

Requisitos Previos
- Python 3.9 o superior instalado.
- Docker (opcional, si deseas ejecutar la aplicación dentro de un contenedor).
- GPU compatible con CUDA (opcional, para optimizar el rendimiento).

## Instalación
1. Clonar el Repositorio

````
git clone https://github.com/tu_usuario/tu_repositorio.git
cd tu_repositorio
````

2. Crear un Entorno Virtual
````
python -m venv env
source env/bin/activate  # En Linux/Mac
env\Scripts\activate     # En Windows
````

3. Instalar Dependencias
````
pip install -r requirements.txt
````

4. Descargar el Modelo
Ejecuta el script para descargar el modelo Stable Diffusion:

````
python descarga_modelo.py
````

- Uso Local (Sin Docker)
    1. Ejecuta la aplicación:
    ````
    streamlit run app.py
    ````

    2. Abre el navegador y accede a: http://localhost:8501.
    
- Uso con Docker
    1. Construir la imagen Docker:

    ````
    docker build -t streamlit-huggingface-app .
    ````
    2. Ejecutar el contenedor:

    ````
    docker run -p 8501:8501 -v $(pwd)/models:/models streamlit-huggingface-app
    ````

    3. Accede a http://localhost:8501 en tu navegador.

## Estructura del Proyecto

````
├── app.py               # Código principal de la aplicación Streamlit
├── utils.py             # Funciones auxiliares (guardar archivos, clasificar imágenes, etc.)
├── descarga_modelo.py   # Script para descargar el modelo de generación
├── requirements.txt     # Dependencias del proyecto
├── Dockerfile           # Configuración para el contenedor Docker
├── models/              # Carpeta donde se almacenan los modelos (descargados)
├── temp/                # Carpeta temporal para archivos subidos

````