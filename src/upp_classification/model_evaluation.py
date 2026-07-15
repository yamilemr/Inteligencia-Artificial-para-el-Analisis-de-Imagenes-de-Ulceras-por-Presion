import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from matplotlib.colors import LinearSegmentedColormap
from upp_classification.config import CLASS_NAMES


def get_predictions(model, dataset):
    """
    Obtiene las etiquetas reales, predichas y las probabilidades del modelo

    Parámetros:
    - model (keras.Model): Modelo entrenado.
    - dataset (tf.data.Dataset): Dataset sobre el que se evaluará el modelo.
    
    Returns:
    - y_true (np.ndarray): Etiquetas reales.
    - y_pred (np.ndarray): Etiquetas predichas por el modelo.
    - y_prob (np.ndarray): Probabilidades predichas por el modelo.
    """
    # Listas para almacenar las etiquetas reales, predicciones y probabilidades
    y_true = []
    y_pred = []
    y_prob = []

    # Recorrer el dataset una sola vez lote por lote para mantener la correspondencia
    # entre las imágenes y sus etiquetas, aún cuando el dataset utiliza shuffle
    for images, labels in dataset:
        # Obtener las probabilidades predichas por el modelo para el lote actual
        probs = model(images, training=False).numpy()

        # Obtener la clase predicha (la de mayor probabilidad) para cada imagen
        preds = np.argmax(probs, axis=1)

        # Almacenar las etiquetas reales, las predicciones y las probabilidades
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)

    # Convertir las listas a arreglos de NumPy
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def calculate_metrics(y_true, y_pred):
    """
    Calcula la matriz de confusión y las métricas globales de evaluación para un problema
    de clasificación multiclase (accuracy, precision, recall y F1-score).

    Parámetros:
    - y_true (np.ndarray): Etiquetas reales.
    - y_pred (np.ndarray): Etiquetas predichas por el modelo.

    Returns:
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


def plot_confusion_matrix(cm, metrics, title):
    """
    Grafica la matriz de confusión e incluye las métricas globales.

    Parámetros:
    - cm (np.ndarray): Matriz de confusión.
    - metrics (dict): Diccionario con las métricas globales.
    - title (str): Título de la figura.

    Returns:
    - None: La función genera la gráfica de la matriz de confusión.
    """
    # Obtener las métricas
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    # Definir un mapa de colores personalizado
    colors = ["#e7eeff", "#8882d9", "#264f73"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_bupu", colors)

    # Inicializar la figura
    plt.figure(figsize=(6, 6))

    # Graficar la matriz de confusión
    sns.heatmap(cm, annot=True, fmt="d", cmap=custom_cmap,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 13})

    # Configurar el título y las etiquetas de los ejes
    plt.title(title, fontsize=14)
    plt.xlabel("Clase predicha", fontsize=12)
    plt.ylabel("Clase real", fontsize=12)

    # Mostrar las métricas debajo de la figura
    plt.figtext(0.45, 0, f"\nAccuracy: {accuracy:.4f} | Precision: {precision:.4f}\nRecall: {recall:.4f} | F1-score: {f1:.4f}",
                ha="center", fontsize=11, va="top")

    # Ajustar el espaciado y mostrar la figura
    plt.tight_layout()
    plt.show()


def calculate_class_metrics(y_true, y_pred):
    """
    Calcula precision, recall y F1-score para cada clase.

    Parámetros:
    - y_true (np.ndarray): Etiquetas reales.
    - y_pred (np.ndarray): Etiquetas predichas por el modelo.

    Retruns:
    - class_metrics (dict): Diccionario con las métricas por clase.
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


def plot_class_metrics(class_metrics, title):
    """
    Grafica las métricas de evaluación por clase en formato de tabla.

    Parámetros:
    - class_metrics (dict): Diccionario con las métricas por clase.
    - title (str): Título de la figura.

    Returns:
    - None: La función genera una tabla con las métricas por clase para un conjunto de datos.
    """
    # Crear DataFrame para la visualización
    class_metrics_df = pd.DataFrame({
        "Precision": class_metrics["precision"],
        "Recall": class_metrics["recall"],
        "F1-score": class_metrics["f1"]
    }, index=CLASS_NAMES)

    # Formatear las anotaciones y resaltar en negritas las métricas perfectas
    annotations = class_metrics_df.T.apply(
        lambda column: column.map(
            lambda value: (
                r"$\bf{1.00}$"
                if np.isclose(value, 1.0)
                else f"{value:.2f}"
            )
        )
    )

    # Inicializar la figura
    plt.figure(figsize=(6, 2.8))

    ax = sns.heatmap(class_metrics_df.T, annot=annotations, fmt="", cmap="Greys", vmin=1, vmax=1,
                     linewidths=0.5, linecolor="#3d3d3d", cbar=False, annot_kws={"size": 12, "color": "#3d3d3d"})

    # Configurar el título y las etiquetas de los ejes
    ax.set_title(title, fontsize=14, color="#3d3d3d")
    ax.set_xticklabels(ax.get_xticklabels(), color="#3d3d3d")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#3d3d3d")
    ax.tick_params(labelsize=10, color="#3d3d3d")

    # Ajustar el espaciado y mostrar la figura
    plt.tight_layout()
    plt.show()


def plot_training_curves(history):
    """
    Grafica las curvas de entrenamiento y validación para accuracy y loss.

    Parámetros:
    - history (keras.callbacks.History): Objeto history devuelto por `model.fit()`.

    Returns:
    - None: La función genera una figura con las curvas de accuracy y loss para train y val.
    """
    # Definir los colores de las curvas
    color_train = "#6e5db0"
    color_val = "#319491"

    # Diccionario con el historial del entrenamiento
    history = history.history

    # Obtener las métricas del historial
    train_acc = history["accuracy"]
    val_acc = history["val_accuracy"]

    train_loss = history["loss"]
    val_loss = history["val_loss"]

    # Definir el rango de épocas y las posiciones de las marcas del eje x
    epochs_range = range(1, len(train_acc) + 1)
    ticks = [1] + list(range(10, len(train_acc) + 1, 10))

    # Agregar la última época si no coincide con una marca existente
    if ticks[-1] != len(train_acc):
        ticks.append(len(train_acc))

    # Crear la figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # Curvas de accuracy
    ax1.plot(epochs_range, train_acc, label="Entrenamiento", color=color_train)
    ax1.plot(epochs_range, val_acc, label="Validación", color=color_val)
    ax1.set_ylabel("Accuracy", fontsize=17, color="#3d3d3d")

    # Curvas de loss
    ax2.plot(epochs_range, train_loss, label="Entrenamiento", color=color_train)
    ax2.plot(epochs_range, val_loss, label="Validación", color=color_val)
    ax2.set_ylabel("Loss", fontsize=17, color="#3d3d3d")

    # Aplicar el mismo formato a ambos subplots
    for ax in (ax1, ax2):
        ax.set_xlabel("Epochs", fontsize=17, color="#3d3d3d")
        ax.set_xticks(ticks)
        ax.tick_params(axis="both", labelsize=11, colors="#3d3d3d")
        ax.legend(fontsize=13, labelcolor="#3d3d3d")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("#3d3d3d")
        ax.spines["bottom"].set_color("#3d3d3d")

    # Ajustar el espaciado y mostrar la figura
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2) 
    plt.show()
