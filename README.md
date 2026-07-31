# Inteligencia Artificial para el Análisis de Imágenes de Úlceras por Presión

**Alumna:** Yamile Montecinos Rodríguez  
**Tutora:** Dra. Lorena Díaz González  

Este repositorio contiene el código fuente del proyecto de tesis para la Licenciatura en Inteligencia Artificial de la Universidad Autónoma del Estado de Morelos.  

El proyecto se enfoca en el desarrollo y la optimización de modelos de aprendizaje profundo para clasificar imágenes de úlceras por presión en las cinco categorías del sistema internacional NPUAP-EPUAP (estadios I, II, III, IV y no estadiable), incorporando una sexta clase correspondiente a la piel sana.  


## Estructura del proyecto
```text
upp_classification/
├── data/                     # Datos crudos y metadatos generados
├── experiments/              # Experimentos de Optuna y MLflow
├── models/                   # Modelos finales exportados (.keras)
├── reports/                  # Gráficas de resultados y métricas
├── scripts/                  # Scripts de ejecución
│   ├── generate_metadata.py                # Genera los metadatos y particiones de los datasets
│   ├── run_hyperparameter_optimization.py  # Ejecuta la búsqueda de hiperparámetros con Optuna
│   └── train_best_model.py                 # Entrena y evalúa el modelo con la mejor configuración
├── src/                      # Código fuente principal del paquete modularizado
│   └── upp_classification/
│       ├── config.py                       # Configuración global del proyecto
│       ├── metadata_generation.py          # Generación de metadatos y particiones de los datasets
│       ├── data_loader.py                  # Carga y preprocesamiento de los datos
│       ├── model_builder.py                # Construcción del modelo de clasificación
│       ├── training.py                     # Lógica de entrenamiento del modelo
│       ├── evaluation.py                   # Evaluación y cálculo de métricas
│       ├── visualization.py                # Generación de gráficas y visualizaciones
│       ├── mlflow_tracking.py              # Registro de experimentos con MLflow
│       └── hyperparameter_optimization.py  # Optimización de hiperparámetros con Optuna
└── pyproject.toml / uv.lock   # Dependencias gestionadas con 'uv'
```


## Instalación y Configuración
El proyecto utiliza uv como gestor de paquetes y dependencias

1. **Clonar el repositorio:**
```bash
git clone https://github.com/yamilemr/Inteligencia-Artificial-para-el-Analisis-de-Imagenes-de-Ulceras-por-Presion.git
cd upp_classification
```

2. **Instalar uv (en caso de que no se tenga):**
```bash
pip install uv
```

3. **Instalar dependencias:**
```bash
# Esto leerá el archivo pyproject.toml / uv.lock y creará el entorno virtual
uv sync
```


## Ejecución
Todos los scripts principales deben ejecutarse desde la raíz del proyecto usando uv run para garantizar que utilicen el entorno virtual correcto.

1. **Generar los metadatos de los datasets:**  
Genera los archivos CSV con los metadatos de los datasets UPP y PIID, incluyendo la partición de las imágenes en los conjuntos de entrenamiento, validación y prueba.
```bash
uv run python scripts/generate_metadata.py
```

2. **Ejecutar la optimización de hiperparámetros:**  
Ejecuta la optimización de hiperparámetros para los modelos de transfer learning utilizando Optuna. Cada configuración evaluada se registra automáticamente en MLflow.  
Los modelos disponibles para realizar la optimización son ResNet50V2, InceptionResNetV2, DenseNet121 y ConvNeXtTiny.
```bash
uv run python scripts/run_hyperparameter_optimization.py --model ResNet50V2 --trials 60 --experiment upp_classification
```

3. **Entrenar el mejor modelo:**  
Entrena y evalúa el modelo de transfer learning utilizando la mejor configuración de hiperparámetros obtenida durante la optimización. El entrenamiento y las métricas también se registran en MLflow, y el modelo final se guarda en el directorio `models/`.
```bash
uv run python scripts/train_best_model.py --model ResNet50V2 --experiment upp_classification
```


## Monitoreo con MLflow
El proyecto utiliza MLflow para registrar y visualizar la información de cada experimento de entrenamiento, incluyendo los hiperparámetros evaluados, las métricas por época, las matrices de confusión, las métricas globales y por clase y los modelos entrenados.  

Para iniciar la interfaz web de MLflow, ejecuta el siguiente comando desde la raíz del proyecto:
```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlflow_tracking.db
```

Una vez iniciado el servidor, accede a la interfaz desde un navegador:
```text
http://127.0.0.1:5000
```


## Contacto
- LinkedIn: [yamilemontecinos](https://www.linkedin.com/in/yamilemontecinos/) 
- Correo electrónico: yamile.montecinos@uaem.edu.mx
