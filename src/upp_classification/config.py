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

# Clases con sus identificadores y nombres descriptivos
CLASS_LABELS  = {
    "ps": {"id": 0,
           "name": "Piel sana"},
    "i": {"id": 1,
          "name": "Estadio I"},
    "ii": {"id": 2,
           "name": "Estadio II"},
    "iii": {"id": 3,
            "name": "Estadio III"},
    "iv": {"id": 4,
           "name": "Estadio IV"},
    "nc": {"id": 5,
           "name": "No estadiable"}
}

# Diccionario para mapear de etiquetas de texto a índices numéricos
LABEL_MAP = {key: value["id"] for key, value in CLASS_LABELS .items()}

# Lista con los nombres descriptivos de las clases
CLASS_NAMES = [value["name"] for value in CLASS_LABELS .values()]