import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from upp_classification.config import CLASS_NAMES, LABEL_TO_NAME, UPP_CSV_FILE, PIID_CSV_FILE


def plot_training_curves(history, title, tick_step=10):
    """
    Genera un gráfico con las curvas de entrenamiento y validación para accuracy y loss.

    Args:
    - history (keras.callbacks.History o dict): Objeto history devuelto por `model.fit()`,
                                                o un diccionario con las métricas.
    - title (str): Título principal para la figura.
    - tick_step (int, optional): Intervalo para las marcas del eje X. Por defecto es 10.

    Returns:
    - matplotlib.figure.Figure: Figura con las curvas de accuracy y loss.
    """
    # Si es un objeto de Keras, extraer el diccionario
    # Si ya es un diccionario, usarlo directamente
    history_dict = history.history if hasattr(history, "history") else history

    # Obtener las métricas del historial
    train_acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]

    train_loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]

    # Definir el rango de épocas y las posiciones de las marcas del eje x
    epochs_range = range(1, len(train_acc) + 1)
    ticks = [1] + list(range(tick_step, len(train_acc) + 1, tick_step))

    # Agregar la última época si no coincide con una marca existente
    if ticks[-1] != len(train_acc):
        ticks.append(len(train_acc))

    # Definir los colores de las curvas
    color_train = "#6856ad"
    color_val = "#279fb4"

    # Crear la figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.04, color="#3d3d3d")

    # Curvas de accuracy
    ax1.plot(epochs_range, train_acc, label="Entrenamiento", linewidth=2, solid_capstyle="round", color=color_train)
    ax1.plot(epochs_range, val_acc, label="Validación", linewidth=2, solid_capstyle="round", color=color_val)
    ax1.set_ylabel("Accuracy", fontsize=14, color="#3d3d3d")

    # Curvas de loss
    ax2.plot(epochs_range, train_loss, label="Entrenamiento", linewidth=2, solid_capstyle="round", color=color_train)
    ax2.plot(epochs_range, val_loss, label="Validación", linewidth=2, solid_capstyle="round", color=color_val)
    ax2.set_ylabel("Loss", fontsize=14, color="#3d3d3d")

    # Aplicar el mismo formato a ambos subplots
    for ax in (ax1, ax2):
        ax.set_xlabel("Épocas", fontsize=12, color="#3d3d3d")
        ax.set_xticks(ticks)
        ax.tick_params(axis="both", length=0, pad=7, colors="#3d3d3d")
        ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.27)
        ax.grid(axis="x", linestyle="-", linewidth=0.5, alpha=0.17)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_color("#9b9b9b")
        ax.spines["bottom"].set_color("#9b9b9b")

    # Leyenda única para ambas gráficas
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), 
               ncol=2, frameon=False, fontsize=12, labelcolor="#3d3d3d")

    # Ajustar el espaciado
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.17)

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


def style_bar_plot(ax, title, ylabel, title_x=0.5, labelrotation_x=0):
    """
    Aplica un estilo visual consistente a una gráfica de barras.

    Args:
    - ax (matplotlib.axes.Axes): Ejes de la gráfica que se desea configurar.
    - title (str): Título de la gráfica.
    - ylabel (str): Etiqueta del eje Y.
    - title_x (float, optional): Coordenada X para centrar el título (0.0 a 1.0). Por defecto es 0.5.
    - labelrotation_x (int, optional): Rotación de las etiquetas del eje X. Por defecto es 0.

    Returns:
    - None: Modifica directamente los ejes (ax) proporcionados.
    """
    # Título y etiquetas de los ejes
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, labelpad=8, color="#3d3d3d")
    ax.set_title(title, fontweight="bold", pad=15, x=title_x, color="#3d3d3d") 

    # Configurar los bordes de la gráfica
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#9b9b9b")

    # Configurar el estilo de los ejes
    ax.tick_params(axis="x", length=0, pad=8, labelrotation=labelrotation_x, colors="#3d3d3d")
    ax.tick_params(axis="y", length=0, colors="#3d3d3d")

    # Grid horizontal
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.xaxis.grid(False)

    # Colocar la cantidad exacta arriba de cada barra
    for container in ax.containers:
        ax.bar_label(container=container, fmt="%d", fontsize=8.5, color="#3d3d3d")


def plot_class_counts_per_dataset(upp_csv_file=UPP_CSV_FILE, piid_csv_file=PIID_CSV_FILE):
    """
    Genera dos gráficos de barras con el número de imágenes por clase para la base de datos principal
    y para el conjunto PIID reclasificado.

    Args:
    - upp_csv_file (str o Path, optional): Ruta del archivo CSV que contiene los metadatos de las 
                                           imágenes del dataset principal. Por defecto es UPP_CSV_FILE.
    - piid_csv_file (str o Path, optional): Ruta del archivo CSV que contiene los metadatos de las 
                                            imágenes del dataset PIID. Por defecto es PIID_CSV_FILE.

    Returns:
    - tuple: Una tupla con las figuras de las distribuciones de cada base de datos:
             - fig_upp (matplotlib.figure.Figure)
             - fig_piid (matplotlib.figure.Figure)
    """
    df_upp = pd.read_csv(upp_csv_file)
    df_piid = pd.read_csv(piid_csv_file)

    # Orden de los nombres de las clases
    class_order_upp = ["Piel sana", "Estadio I", "Estadio II", "Estadio III", "Estadio IV", "No estadiable"]
    class_order_piid = class_order_upp[1:]

    # Contar imágenes por clase
    counts_upp = df_upp["label"].map(LABEL_TO_NAME).value_counts().reindex(index=class_order_upp)
    counts_piid = df_piid["label"].map(LABEL_TO_NAME).value_counts().reindex(index=class_order_piid)

    # Gráfica de la base de datos principal
    fig_upp, ax_upp = plt.subplots(figsize=(5.7, 3.8))
    ax_upp.bar(counts_upp.index, counts_upp.values, width=0.5, color="#889ab9")
    style_bar_plot(
        ax=ax_upp, 
        title="Imágenes por clase de la base de datos principal\n", 
        ylabel="Número de imágenes", 
        title_x=0.44,
        labelrotation_x=27)
    ax_upp.text(0.44, 1.07, f"Total = {counts_upp.sum()} imágenes", 
                transform=ax_upp.transAxes, ha="center", va="bottom", color="#3d3d3d")
    fig_upp.tight_layout()

    # Gráfica de PIID
    fig_piid, ax_piid = plt.subplots(figsize=(4.7, 3.8))
    ax_piid.bar(counts_piid.index, counts_piid.values, width=0.5, color="#889ab9")
    style_bar_plot(
        ax=ax_piid,
        title="Imágenes por clase del conjunto PIID\n",
        ylabel="Número de imágenes",
        title_x=0.44,
        labelrotation_x=27
    )
    ax_piid.text(0.44, 1.07, f"Total = {counts_piid.sum()} imágenes", 
                 transform=ax_piid.transAxes, ha="center", va="bottom", color="#3d3d3d")
    fig_piid.tight_layout()

    return fig_upp, fig_piid


def plot_class_split_distribution(upp_csv_file=UPP_CSV_FILE, piid_csv_file=PIID_CSV_FILE):
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

    # Agregar columnas faltantes en PIID para poder concatenar ambos datasets
    df_piid["patient_id"] = None
    df_piid["lesion_id"] = None
    df_all = pd.concat([df_upp, df_piid], ignore_index=True)

    # Cambiar los nombres de las clases y de los splits
    split_map = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}

    class_order = ["Piel sana", "Estadio I", "Estadio II", "Estadio III", "Estadio IV", "No estadiable"]
    split_order = ["Entrenamiento", "Validación", "Prueba"]

    # Aplicar el mapeo a la información de los datasets combinados
    label_plot = df_all["label"].map(LABEL_TO_NAME)
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
    ax = counts.plot(kind="bar", figsize=(7.5, 4.2), width=0.7, color=["#b6cee4", "#95c8cf", "#d2ddbf"]) 
    fig = ax.get_figure()
    
    # Aplicar el estilo de la gráfica
    style_bar_plot(
        ax=ax, 
        title="Distribución de imágenes por clase y partición\n", 
        ylabel="Número de imágenes",
        title_x=0.45,
        labelrotation_x=0
    )

    # Agregar el número total de imágenes
    ax.text(0.45, 1.05, f"Total = {split_totals.sum()} imágenes", 
            transform=ax.transAxes, ha="center", va="bottom", color="#3d3d3d")
    
    # Configurar el estilo de la leyenda
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3, frameon=False)
    for text in legend.get_texts():
        text.set_color("#3d3d3d")

    # Ajustar el espaciado
    fig.tight_layout()

    return fig


def plot_patient_split_assignment(upp_csv_file=UPP_CSV_FILE):
    """
    Genera una matriz que muestra la asignación de cada paciente a una única partición.

    Args:
    - upp_csv_file (str o Path, optional): Ruta del archivo CSV que contiene los metadatos de las 
                                            imágenes del dataset principal. Por defecto es UPP_CSV_FILE.

    Returns:
    - matplotlib.figure.Figure: Figura con la asignación de pacientes por split.
    """
    df = pd.read_csv(upp_csv_file)

    # Filtrar, eliminar duplicados y ordenar alfabéticamente (p001 a p259)
    patients = df[["patient_id", "split"]].dropna().drop_duplicates().sort_values("patient_id")

    # Diccionarios de mapeo para los colores y las etiquetas de cada split
    split_colors = {"train": "#9abfde", "val": "#68c5d0", "test": "#c7d7ab"}
    split_names = {"train": "Entrenamiento", "val": "Validación", "test": "Prueba"}
    
    # Crear la matriz paciente x split usando Pandas Pivot
    patients["val"] = patients["split"].map({"train": 1, "val": 2, "test": 3})
    pivot = (
        patients.pivot(index="split", columns="patient_id", values="val")
        .reindex(["train", "val", "test"])
        .fillna(0)
    )
    patient_order = pivot.columns.tolist()

    # Mapa de colores
    cmap = LinearSegmentedColormap.from_list("split_colors", ["#e5e5e5", *split_colors.values()], N=4)

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=3)

    # Título
    ax.set_title("Asignación de los pacientes a cada partición\n", 
                 fontweight="bold", fontsize=14, pad=12, x=0.45, color="#3d3d3d")

    # Agregar el número total de pacientes
    ax.text(0.45, 1.05, f"Total = {len(patient_order)} pacientes", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12, color="#3d3d3d")

    # Líneas separadoras con grid
    ax.set_xticks(np.arange(-0.5, len(patient_order)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, axis="x")
    
    # Etiquetas del eje X
    ticks = list(range(0, len(patient_order), 20))
    if ticks[-1] != len(patient_order) - 1:
        ticks.append(len(patient_order) - 1)
        
    ax.set_xticks(ticks)
    ax.set_xticklabels([patient_order[i] for i in ticks], fontsize=10)
    ax.tick_params(axis="x", which="both", length=0, pad=8, colors="#3d3d3d")
    ax.set_xlabel("Pacientes", labelpad=8, fontsize=12, color="#3d3d3d")

    # Etiquetas del eje Y
    ax.set_yticks(range(3))
    ax.set_yticklabels([split_names[s] for s in ["train", "val", "test"]], fontsize=12)
    ax.tick_params(axis="y", which="both", length=0, colors="#3d3d3d")

    # Eliminar bordes
    for spine in ax.spines.values(): 
        spine.set_visible(False)

    # Leyenda
    counts = patients["split"].value_counts()
    legend_handles = [
        plt.Line2D(
            [0], [0], 
            marker="s", 
            color="w", 
            markerfacecolor=split_colors[k], 
            markersize=9, 
            label=f"{split_names[k]} ({counts.get(k, 0)} pacientes)"
        )
        for k in ["train", "val", "test"]
    ]
    legend = ax.legend(handles=legend_handles, loc="upper center", fontsize=12,
                       bbox_to_anchor=(0.45, -0.25), ncol=3, frameon=False)
    for text in legend.get_texts():
            text.set_color("#3d3d3d")

    # Ajustar el espaciado
    plt.tight_layout()
    
    return fig
