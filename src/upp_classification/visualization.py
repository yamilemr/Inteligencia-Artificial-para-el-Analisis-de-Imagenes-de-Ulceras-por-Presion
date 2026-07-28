import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from upp_classification.config import CLASS_NAMES, CLASS_LABELS, UPP_CSV_FILE, PIID_CSV_FILE


def plot_training_curves(history):
    """
    Genera un gráfico con las curvas de entrenamiento y validación para accuracy y loss.

    Args:
    - history (keras.callbacks.History): Objeto history devuelto por `model.fit()`.

    Returns:
    - matplotlib.figure.Figure: Figura con las curvas de accuracy y loss.
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

    # Ajustar el espaciado
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2) 

    return fig


def plot_confusion_matrix(cm, metrics, title, class_names=CLASS_NAMES):
    """
    Genera un gráfico con la matriz de confusión (incluye las métricas globales).

    Args:
    - cm (np.ndarray): Matriz de confusión.
    - metrics (dict): Diccionario con las métricas globales.
    - title (str): Título de la figura.
    - class_names (list[str], optional): Lista con los nombres de las clases en el mismo orden que 
                                         sus identificadores numéricos. Por defecto es CLASS_NAMES.

    Returns:
    - matplotlib.figure.Figure: Figura de la matriz de confusión.
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
    fig, ax = plt.subplots(figsize=(6, 6))

    # Graficar la matriz de confusión
    sns.heatmap(cm, annot=True, fmt="d", cmap=custom_cmap,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 13}, ax=ax)

    # Configurar el título y las etiquetas de los ejes
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Clase predicha", fontsize=12)
    ax.set_ylabel("Clase real", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.text(0.45, -0.4, f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f}\nRecall: {recall:.4f} | F1-score: {f1:.4f}",
            ha="center", va="top", transform=ax.transAxes, fontsize=11)

    # Ajustar el espaciado
    plt.tight_layout()
    
    return fig


def plot_class_metrics(class_metrics, title, class_names=CLASS_NAMES):
    """
    Genera un gráfico con las métricas de evaluación por clase en formato de tabla.

    Args:
    - class_metrics (dict): Diccionario con las métricas por clase.
    - title (str): Título de la figura.
    - class_names (list[str], optional): Lista con los nombres de las clases en el mismo orden que 
                                         sus identificadores numéricos. Por defecto es CLASS_NAMES.

    Returns:
    - matplotlib.figure.Figure: Figura con la tabla de métricas por clase.
    """
    # Crear DataFrame para la visualización
    class_metrics_df = pd.DataFrame({
        "Precision": class_metrics["precision"],
        "Recall": class_metrics["recall"],
        "F1-score": class_metrics["f1"]
    }, index=class_names)

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
    fig, ax = plt.subplots(figsize=(6, 2.8))

    sns.heatmap(class_metrics_df.T, annot=annotations, fmt="", cmap="Greys", vmin=1, vmax=1, linewidths=0.5,
                linecolor="#3d3d3d", cbar=False, annot_kws={"size": 12, "color": "#3d3d3d"}, ax=ax)

    # Configurar el título y las etiquetas de los ejes
    ax.set_title(title, fontsize=14, color="#3d3d3d")
    ax.set_xticklabels(ax.get_xticklabels(), color="#3d3d3d")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, color="#3d3d3d")
    ax.tick_params(labelsize=10, color="#3d3d3d")

    # Ajustar el espaciado y mostrar la figura
    plt.tight_layout()
    
    return fig


def plot_dataset_distribution(upp_csv_file=UPP_CSV_FILE, piid_csv_file=PIID_CSV_FILE):
    """
    Genera un gráfico de barras con la distribución de imágenes por clase y split (train, val y test)
    de ambos conjuntos (UPP y PIID).

    Args:
    - upp_csv_file (str o Path, optional): Ruta del archivo CSV que contiene los metadatos de las 
                                           imágenes del dataset principal. Por defecto es UPP_CSV_FILE.
    - piid_csv_file (str o Path, optional): Ruta del archivo CSV que contiene los metadatos de las 
                                            imágenes del dataset PIID. Por defecto es PIID_CSV_FILE.

    Returns:
    - matplotlib.figure.Figure: Figura con la gráfica de distribución generada.
    """
    df_upp = pd.read_csv(upp_csv_file)
    df_piid = pd.read_csv(piid_csv_file)

    # Alinear columnas faltantes en PIID para poder concatenar ambos datasets
    df_piid["patient_id"] = None
    df_piid["lesion_id"] = None
    df_all = pd.concat([df_upp, df_piid], ignore_index=True)

    # Cambiar los nombres de las clases y de los splits
    label_map = {key: value["name"] for key, value in CLASS_LABELS.items()}
    split_map = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}

    class_order = ["Piel sana", "Estadio I", "Estadio II", "Estadio III", "Estadio IV", "No estadiable"]
    split_order = ["Entrenamiento", "Validación", "Prueba"]

    # Aplicar el mapeo a la información de los datasets combinados
    label_plot = df_all["label"].map(label_map)
    split_plot = df_all["split"].map(split_map)

    # Contar imágenes por clase y por split
    counts = (
        pd.DataFrame({"label": label_plot, "split": split_plot})
        .groupby(["label", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=class_order, columns=split_order)
    )

    # Número total de imágenes para cada split
    split_totals = counts.sum(axis=0)
    counts.columns = [f"{col} ({split_totals[col]} imgs)" for col in counts.columns]

    # Graficar
    ax = counts.plot(kind="bar", figsize=(7, 4), width=0.7, color=["#8c74b5", "#95a8cf", "#b6cee4"])
    
    ax.set_xlabel("")
    ax.set_title("Distribución de imágenes por clase y split")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colocar la cantidad exacta arriba de cada barra
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=8.5)

    # Ajustar el espaciado y la rotación de los nombres en el eje x
    plt.xticks(rotation=0)
    plt.tight_layout()

    return ax.get_figure()
