import os
from pathlib import Path
from dotenv import load_dotenv

# Rutas que provienen del .env
load_dotenv()

UPP_IMGS_DIR = Path(os.environ["UPP_IMGS_DIR"]) # Carpeta que contiene las imágenes del dataset principal
PIID_IMGS_DIR = Path(os.environ["PIID_IMGS_DIR"]) # Carpeta que contiene las imágenes reclasificadas de PIID

# Rutas del repositorio
BASE_DIR = Path(__file__).resolve().parents[2] # Encuentra automáticamente la ruta absoluta de la carpeta donde está este config.py

UPP_CSV_FILE = BASE_DIR / "data" / "labels_upp.csv"
PIID_CSV_FILE = BASE_DIR / "data" / "labels_piid.csv"

# Constantes
IMAGE_SIZE = (224, 224)

LABELS = {
    "ps": 0,
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "nc": 5
}