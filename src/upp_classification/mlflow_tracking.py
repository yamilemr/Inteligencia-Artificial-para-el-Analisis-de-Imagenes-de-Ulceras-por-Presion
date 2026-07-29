import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from upp_classification.visualization import plot_confusion_matrix, plot_class_metrics
from upp_classification.config import MLFLOW_TRACKING_URI, MLFLOW_ARTIFACTS_DIR, MODELS_DIR, CLASS_LABELS


def setup_mlflow(experiment_name, tracking_uri=MLFLOW_TRACKING_URI, artifacts_dir=MLFLOW_ARTIFACTS_DIR):
    """
    Configura el servidor de tracking y el experimento activo de MLflow.

    Args:
    - experiment_name (str): Nombre del experimento en el que se registrarán los runs de entrenamiento.
    - tracking_uri (str, optional): URI del backend de tracking utilizado por MLflow para almacenar los
                                    experimentos. Por defecto es MLFLOW_TRACKING_URI.
    - artifacts_dir (str o Path, optional): Directorio donde se almacenarán los artefactos asociados a
                                            los runs. Por defecto es MLFLOW_ARTIFACTS_DIR.

    Returns:
    - None: La función configura el experimento activo de MLflow.
    """
    # Configurar el backend de tracking utilizado por MLflow para almacenar la información de los
    # experimentos (runs, parámetros, métricas y tags)
    mlflow.set_tracking_uri(uri=tracking_uri)

    # Comprobar si el experimento ya existe
    experiment = mlflow.get_experiment_by_name(name=experiment_name)

    # Si el experimento no existe, crearlo especificando la ubicación donde se almacenarán los 
    # artefactos generados por los runs del experimento
    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifacts_dir.resolve().as_uri()
        )

    # Establecer el experimento como activo para que todos los runs iniciados después se registren ahí
    mlflow.set_experiment(experiment_name=experiment_name)


def generate_run_name(architecture_name, params):
    """
    Genera el nombre del run de MLflow a partir de la arquitectura y los hiperparámetros
    utilizados durante el entrenamiento.

    El nombre sigue el formato:
    ConvNeXtTiny | dl:2, du1:32, du2:64, dr:0.2, opt:Adam, lr:1e-4, bs:32
    ConvNeXtTiny | dl:2, du1:32, du2:64, dr:0.2, opt:AdamW, lr:1e-4, wd:1e-6, bs:32

    Args:
    - architecture_name (str): Nombre de la arquitectura utilizada (ej. ConvNeXtTiny).
    - params (dict): Diccionario con los hiperparámetros seleccionados para el entrenamiento.
                     Debe contener:
                     - dense_layers (int): Número de capas densas.
                     - dense_units (list[int]): Número de neuronas por capa densa.
                     - dropout_rate (float): Tasa de dropout aplicada después de cada capa densa
                                             o directamente después del GlobalAveragePooling2D si
                                             no existen capas densas.
                     - optimizer (str): Optimizador.
                     - learning_rate (float): Tasa de aprendizaje del optimizador.
                     - weight_decay (float): Decaimiento de pesos (sólo para AdamW).
                     - batch_size (int): Tamaño de batch.

    Returns:
    - str: Nombre descriptivo del run para su registro en MLflow.
    """
    hyperparams = []
    
    # Número de capas densas
    dense_layers = params["dense_layers"]
    hyperparams.append(f"dl:{dense_layers}")
    
    # Número de neuronas en cada capa densa (solo si hay capas densas)
    if dense_layers > 0:
        for i, units in enumerate(params["dense_units"]):
            hyperparams.append(f"du{i+1}:{units}")
            
    # Tasa de dropout
    hyperparams.append(f"dr:{params['dropout_rate']}")
    
    # Optimizador
    optimizer = params["optimizer"]
    hyperparams.append(f"opt:{optimizer}")

    # Learning rate
    hyperparams.append(f"lr:{params['learning_rate']}")
    
    # Se agrega weight_decay sólo para AdamW
    if optimizer == "AdamW":
        hyperparams.append(f"wd:{params['weight_decay']}")
            
    # Batch size
    hyperparams.append(f"bs:{params['batch_size']}")

    # Se crea el run_name con el formato especificado
    run_name = f"{architecture_name} | " + ", ".join(hyperparams)

    return run_name


def log_params_to_mlflow(params):
    """
    Registra en MLflow los hiperparámetros utilizados durante el entrenamiento.

    Args:
    - params (dict): Diccionario con los hiperparámetros del modelo.

    Returns:
    - None: La función registra los hiperparámetros en el run activo de MLflow.
    """
    params_mlflow = params.copy()

    # Expandir la lista dense_units en parámetros individuales (dense_units_1, dense_units_2)
    if "dense_units" in params_mlflow:
        dense_units = params_mlflow.pop("dense_units")

        for i, units in enumerate(dense_units):
            params_mlflow[f"dense_units_{i+1}"] = units
    
    # Eliminar weight_decay si el optimizador seleccionado no es AdamW
    if params_mlflow["optimizer"] != "AdamW":
        params_mlflow.pop("weight_decay", None)
            
    # Registrar los hiperparámetros en el run activo de MLflow
    mlflow.log_params(params=params_mlflow)


class MLflowMetricsCallback(Callback):
    """
    Callback de Keras que registra automáticamente en MLflow las métricas generadas
    al finalizar cada época de entrenamiento.

    Las métricas se almacenan utilizando el número de época como paso (step) para 
    facilitar la visualización de las curvas de entrenamiento.
    """
    def on_epoch_end(self, epoch, logs=None):
        """
        Registra en MLflow las métricas correspondientes a la época actual.

        Args:
        - epoch (int): Índice de la época finalizada.
        - logs (dict, optional): Diccionario con las métricas calculadas por 
                                 Keras durante la época. Por defecto es None.

        Returns:
        - None: La función registra las métricas de la época en MLflow.
        """
        # Inicializar un diccionario vacío si Keras no proporciona métricas
        logs = logs or {}

        # Convertir los valores de las métricas a tipo float
        metrics = {key: float(value) for key, value in logs.items()}

        # Registrar las métricas utilizando la época como step
        mlflow.log_metrics(metrics=metrics, step=epoch)


def log_metrics_to_mlflow(cm, metrics, dataset_name):
    """
    Registra en MLflow las métricas de evaluación y la matriz de confusión de un conjunto de datos.

    Args:
    - cm (np.ndarray): Matriz de confusión.
    - metrics (dict): Diccionario con las métricas globales (accuracy, precision, recall y F1-score).
    - dataset_name (str): Nombre del conjunto de datos evaluado ("train", "val" o "test").

    Returns:
    - None: La función registra las métricas globales y la matriz de confusión dentro del run activo de MLflow.
    """
    # Registrar las métricas de evaluación
    metrics_mlflow = {f"{key}_{dataset_name}": value for key, value in metrics.items()}
    mlflow.log_metrics(metrics=metrics_mlflow)

    # Registrar la matriz de confusión como tabla
    cm_df = pd.DataFrame(cm)
    mlflow.log_table(data=cm_df, artifact_file=f"confusion_matrix/{dataset_name}.json")

    # Registrar la matriz de confusión como imagen (también contiene las métricas)
    fig = plot_confusion_matrix(cm=cm, metrics=metrics, title=f"{dataset_name}")
    mlflow.log_figure(figure=fig, artifact_file=f"confusion_matrix/{dataset_name}.png")
    plt.close(fig)


def log_class_metrics_to_mlflow(class_metrics, dataset_name):
    """
    Registra en MLflow las métricas por clase, también se registra una figura que 
    contiene una tabla con estas métricas.

    Args:
    - class_metrics (dict): Diccionario con precision, recall y f1 por clase.
    - dataset_name (str): Nombre del conjunto de datos evaluado ("train", "val" o "test").

    Returns:
    - None: La función registra las métricas por clase dentro del run activo de MLflow.
    """
    # Etiquetas en el mismo orden que sus identificadores
    class_labels = list(CLASS_LABELS.keys())

    # Registrar las métricas por clase
    class_metrics_mlflow = {}
    for idx, label in enumerate(class_labels):
        class_metrics_mlflow[f"precision_{label}_{dataset_name}"] = float(class_metrics["precision"][idx])
        class_metrics_mlflow[f"recall_{label}_{dataset_name}"] = float(class_metrics["recall"][idx])
        class_metrics_mlflow[f"f1_{label}_{dataset_name}"] = float(class_metrics["f1"][idx])

    mlflow.log_metrics(class_metrics_mlflow)

    # Registrar una imagen que contiene una tabla con las métricas por clase
    fig = plot_class_metrics(class_metrics=class_metrics, title=dataset_name)
    mlflow.log_figure(figure=fig, artifact_file=f"class_metrics/{dataset_name}.png")
    plt.close(fig)


def log_and_save_model(model, architecture_name, models_dir=MODELS_DIR):
    """
    Registra en MLflow el modelo y también lo guarda de forma local.

    Args:
    - model (keras.Model): Modelo entrenado a guardar.
    - architecture_name (str): Nombre de la arquitectura (ej. ConvNeXtTiny).
    - models_dir (Path, optional): Ruta al directorio para guardar el modelo. 
                                   Por defecto es MODELS_DIR.

    Returns:
    - None: La función registra el modelo en MLflow y lo guarda localmente.
    """
    registered_model_name = f"best_{architecture_name}_model"

    # Registrar el modelo de Keras como artefacto en MLflow
    mlflow.tensorflow.log_model(
        model=model, 
        artifact_path="best_model",
        registered_model_name=registered_model_name
    )

    # Guardar localmente
    model.save(models_dir / f"{registered_model_name}.keras")
