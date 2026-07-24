import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from upp_classification.mlflow_tracking import log_metrics_to_mlflow, log_class_metrics_to_mlflow


def get_predictions(model, dataset):
    """
    Obtiene las etiquetas reales, predichas y las probabilidades del modelo.

    Args:
    - model (keras.Model): Modelo entrenado.
    - dataset (tf.data.Dataset): Dataset sobre el que se evaluará el modelo.
    
    Returns:
    - tuple: Una tupla que contiene:
             - y_true (np.ndarray): Etiquetas reales.
             - y_pred (np.ndarray): Etiquetas predichas por el modelo.
             - y_prob (np.ndarray): Probabilidades predichas por el modelo.
    """
    # Listas para almacenar las etiquetas reales, predicciones y probabilidades de cada lote
    y_true_batches = []
    y_pred_batches = []
    y_prob_batches = []

    # Recorrer el dataset una sola vez lote por lote para mantener la correspondencia
    # entre las imágenes y sus etiquetas, aún cuando el dataset utiliza shuffle
    for images, labels in dataset:
        # Obtener los logits predichos por el modelo para el lote actual
        logits = model(images, training=False)
        
        # Obtener la clase predicha y las probabilidades asociadas a cada imagen
        preds = tf.argmax(logits, axis=-1).numpy()
        probs = tf.nn.softmax(logits, axis=-1).numpy()

        # Almacenar las etiquetas reales, predicciones y probabilidades del lote actual
        y_true_batches.append(labels.numpy())
        y_pred_batches.append(preds)
        y_prob_batches.append(probs)

    # Unir todos los lotes en un solo arreglo de NumPy
    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    y_prob = np.concatenate(y_prob_batches, axis=0)

    return y_true, y_pred, y_prob


def calculate_metrics(y_true, y_pred):
    """
    Calcula la matriz de confusión y las métricas globales de evaluación para un problema
    de clasificación multiclase (accuracy, precision, recall y F1-score).

    Args:
    - y_true (np.ndarray): Etiquetas reales.
    - y_pred (np.ndarray): Etiquetas predichas por el modelo.

    Returns:
    - tuple: Una tupla que contiene:
             - cm (np.ndarray): Matriz de confusión.
             - metrics (dict): Diccionario con las métricas globales.
    """
    # Matriz de confusión
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Métricas globales
    metrics = {
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
        "precision": precision_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0)
    }

    return cm, metrics


def calculate_class_metrics(y_true, y_pred):
    """
    Calcula precision, recall y F1-score para cada clase.

    Args:
    - y_true (np.ndarray): Etiquetas reales.
    - y_pred (np.ndarray): Etiquetas predichas por el modelo.

    Retruns:
    - dict: Diccionario con las métricas por clase.
    """
    # Calcular las métricas por clase
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, zero_division=0)

    # Diccionario con las métricas por clase
    class_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    return class_metrics 


def evaluate_model_datasets(model, train_ds=None, val_ds=None, test_ds=None, return_predictions=True, include_class_metrics=True, use_mlflow=True):
    """
    Evalúa un modelo entrenado sobre los datasets proporcionados de forma independiente.
    Permite registrar las métricas globales y matrices de confusión en MLflow si se especifica.

    Args:
    - model (keras.Model): Modelo entrenado a evaluar.
    - train_ds (tf.data.Dataset, optional): Dataset de entrenamiento. Por defecto es None.
    - val_ds (tf.data.Dataset, optional): Dataset de validación. Por defecto es None.
    - test_ds (tf.data.Dataset, optional): Dataset de prueba. Por defecto es None.
    - return_predictions (bool, optional): Indica si se incluyen y_true, y_pred y y_prob en los resultados. Por defecto es True.
    - include_class_metrics (bool, optional): Indica si se calculan las métricas por clase. Por defecto es True.
    - use_mlflow (bool, optional): Indica si se deben registrar las métricas globales y matrices en MLflow. Por defecto es True.

    Returns:
    - dict: Diccionario con las métricas y matrices de confusión calculadas para cada 
            dataset proporcionado. Su estructura es:
            {
                "nombre_dataset": {
                    "y_true": np.ndarray (optional),
                    "y_pred": np.ndarray (optional),
                    "y_prob": np.ndarray (optional),
                    "confusion_matrix": np.ndarray,
                    "metrics": dict,
                    "class_metrics": dict (optional)
                }
            }
    """
    results = {}

    # Mapear los nombres de los conjuntos de datos a sus variables correspondientes
    datasets = {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds
    }

    # Iterar sobre cada conjunto de datos disponible en el diccionario
    for ds_name, dataset in datasets.items():
        # Si no se proporcionó este dataset, se omite la evaluación
        if dataset is None:
            continue
        
        # Diccionario para almacenar los resultados del dataset actual
        dataset_results = {}
        
        # Obtener las etiquetas reales, predichas y las probabilidades
        y_true, y_pred, y_prob = get_predictions(model=model, dataset=dataset)

        # Agregar predicciones si está indicado
        if return_predictions:
            dataset_results.update({
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob
            })
        
        # Calcular la matriz de confusión y las métricas globales
        cm, metrics = calculate_metrics(y_true=y_true, y_pred=y_pred)
        dataset_results.update({
            "confusion_matrix": cm,
            "metrics": metrics
        })

        # Calcular las métricas por clase si está indicado
        if include_class_metrics:
            class_metrics = calculate_class_metrics(y_true=y_true, y_pred=y_pred)
            dataset_results["class_metrics"] = class_metrics

        # Guardar resultados del dataset
        results[ds_name] = dataset_results
        
        # Registrar las métricas de evaluación en MLflow si está indicado
        if use_mlflow:
            log_metrics_to_mlflow(cm=cm, metrics=metrics, dataset_name=ds_name)

            if include_class_metrics:
                log_class_metrics_to_mlflow(class_metrics=class_metrics, dataset_name=ds_name)

    return results
