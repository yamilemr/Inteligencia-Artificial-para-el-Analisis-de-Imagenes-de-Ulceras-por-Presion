import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar las variables de entorno desde el .env
load_dotenv()

UPP_IMGS_DIR = Path(os.getenv("UPP_IMGS_DIR")) # Carpeta que contiene las imágenes del dataset principal
PIID_IMGS_DIR = Path(os.getenv("PIID_IMGS_DIR")) # Carpeta que contiene las imágenes reclasificadas de PIID

# Directorio raíz del repositorio
BASE_DIR = Path(__file__).resolve().parents[2] 

# Archivos CSV con los datos de las imágenes de cada dataset
UPP_CSV_FILE = BASE_DIR / "data" / "labels_upp.csv"
PIID_CSV_FILE = BASE_DIR / "data" / "labels_piid.csv"

# Configuración de las imágenes
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)

# Hiperparámetros
EPOCHS = 100

# Etiquetas
LABELS = {
    "ps": 0,
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "nc": 5
}