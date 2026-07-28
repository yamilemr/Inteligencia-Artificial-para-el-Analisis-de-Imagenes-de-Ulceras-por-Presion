import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar las variables de entorno desde el .env
load_dotenv()

UPP_IMGS_DIR = Path(os.getenv("UPP_IMGS_DIR")) # Carpeta que contiene las imágenes del dataset principal
PIID_IMGS_DIR = Path(os.getenv("PIID_IMGS_DIR")) # Carpeta que contiene las imágenes reclasificadas de PIID

# Directorio raíz del repositorio
BASE_DIR = Path(__file__).resolve().parents[2] 

# Directorio para almacenar los archivos CSV
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Archivos CSV con los metadatos de las imágenes de cada dataset
UPP_CSV_FILE = DATA_DIR / "upp_metadata.csv"
PIID_CSV_FILE = DATA_DIR / "piid_metadata.csv"

# Directorio para almacenar los experimentos (Optuna y MLflow)
EXPERIMENTS_DIR = BASE_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Directorio donde se almacenan los estudios de Optuna
OPTUNA_DIR = EXPERIMENTS_DIR / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

# Directorio donde se almacenan los artefactos de MLflow
MLFLOW_ARTIFACTS_DIR = EXPERIMENTS_DIR / "mlflow_artifacts"
MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Base de datos SQLite utilizada por MLflow para almacenar los experimentos, runs, parámetros y métricas
MLFLOW_TRACKING_URI = f"sqlite:///{EXPERIMENTS_DIR / 'mlflow_tracking.db'}"

# Directorio para almacenar los reportes (gráficas y métricas)
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Directorio donde se almacenan las figuras generadas
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de las imágenes
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)

# Hiperparámetros
EPOCHS = 100

# Semilla global para garantizar la reproducibilidad de las operaciones aleatorias
SEED = 18

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
LABEL_MAP = {key: value["id"] for key, value in CLASS_LABELS.items()}

# Lista con los nombres descriptivos de las clases
CLASS_NAMES = [value["name"] for value in CLASS_LABELS.values()]

# Número de clases
NUM_CLASSES = len(CLASS_LABELS)

# Arquitecturas disponibles para transfer learning
AVAILABLE_MODELS = ["ResNet50V2", "InceptionResNetV2", "DenseNet121", "ConvNeXtTiny"]

# Diccionario con el espacio de búsqueda común a todas las arquitecturas
BASE_SEARCH_SPACE = {
    "dense_layers": {"low": 0, "high": 2},
    "dropout_rate": {"low": 0.2, "high": 0.5, "step": 0.1},
    "optimizer_params": {
        "Adam": {
            "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True}
        },
        "AdamW": {
            "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
            "weight_decay": {"low": 1e-7, "high": 1e-2, "log": True}
        }
    },
    "batch_size": [16, 32]
}

# Espacio de búsqueda específico por arquitectura; dense_units se define según la dimensión del GAP
# de cada modelo base: ResNet50V2 (2048), InceptionResNetV2 (1536), DenseNet121 (1024), ConvNeXtTiny (768)
SEARCH_SPACES = {
    "ResNet50V2": {
        **BASE_SEARCH_SPACE,
        "dense_units": [64, 128, 256, 512]
    },

    "InceptionResNetV2": {
        **BASE_SEARCH_SPACE,
        "dense_units": [64, 128, 256, 512]
    },

    "DenseNet121": {
        **BASE_SEARCH_SPACE,
        "dense_units": [32, 64, 128, 256]
    },

    "ConvNeXtTiny": {
        **BASE_SEARCH_SPACE,
        "dense_units": [32, 64, 128, 256]
    }
}
