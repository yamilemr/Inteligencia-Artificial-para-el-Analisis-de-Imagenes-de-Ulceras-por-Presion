from pathlib import Path

# Directorio raíz del repositorio
BASE_DIR = Path(__file__).resolve().parents[2] 

# Directorio para almacenar los datos
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Carpetas que contienen las imágenes de los datasets
UPP_IMGS_DIR = DATA_DIR / "upp" # Imágenes del dataset principal
PIID_IMGS_DIR = DATA_DIR / "piid_reclassified" # Imágenes reclasificadas de PIID

# Archivos CSV con los metadatos de las imágenes de cada dataset
UPP_CSV_FILE = DATA_DIR / "upp_metadata.csv"
PIID_CSV_FILE = DATA_DIR / "piid_metadata.csv"

# Directorio para almacenar caches de tf.data en disco
CACHE_DIR = DATA_DIR / "tf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

# Directorio para almacenar los modelos entrenados
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Directorio para almacenar los reportes (gráficas y métricas)
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Directorio donde se almacenan las figuras generadas
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

# Diccionario para mapear de etiquetas de texto a nombres descriptivos
LABEL_TO_NAME = {key: value["name"] for key, value in CLASS_LABELS.items()}

# Lista con los nombres descriptivos de las clases
CLASS_NAMES = [value["name"] for value in CLASS_LABELS.values()]

# Número de clases
NUM_CLASSES = len(CLASS_LABELS)

# Modelos disponibles para entrenamiento
AVAILABLE_MODELS = {
    "ResNet50V2": {"image_size": (224, 224)},
    "InceptionResNetV2": {"image_size": (299, 299)},
    "DenseNet121": {"image_size": (224, 224)},
    "ConvNeXtTiny": {"image_size": (224, 224)},
    "CustomCNN": {"image_size": (420, 420)}
}

# Diccionario con el espacio de búsqueda común a todas las arquitecturas preentrenadas
TL_SEARCH_SPACE = {
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

# Espacio de búsqueda específico por arquitectura
SEARCH_SPACES = {
    "ResNet50V2": {
        **TL_SEARCH_SPACE,
        "dense_units": [64, 128, 256, 512]
    },

    "InceptionResNetV2": {
        **TL_SEARCH_SPACE,
        "dense_units": [64, 128, 256, 512]
    },

    "DenseNet121": {
        **TL_SEARCH_SPACE,
        "dense_units": [32, 64, 128, 256]
    },

    "ConvNeXtTiny": {
        **TL_SEARCH_SPACE,
        "dense_units": [32, 64, 128, 256]
    },

    "CustomCNN": {
        "conv_layers": {"low": 1, "high": 3},
        "filters": [16, 32, 64, 128],
        "kernel_size": [3, 5],
        "strides": {"low": 1, "high": 2},
        "dense_layers": {"low": 0, "high": 2},
        "dense_units": [32, 64, 128],
        "dropout_rate": {"low": 0.3, "high": 0.5, "step": 0.1},
        "optimizer_params": {
            "Adam": {
                "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True}
            },
            "AdamW": {
                "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
                "weight_decay": {"low": 1e-6, "high": 1e-3, "log": True}
            }
        },
        "batch_size": [16, 32]
    }
}
