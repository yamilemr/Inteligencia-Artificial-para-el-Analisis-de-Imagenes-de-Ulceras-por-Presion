import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from upp_classification.config import UPP_IMGS_DIR, UPP_CSV_FILE, PIID_IMGS_DIR, PIID_CSV_FILE, LABEL_MAP, IMAGE_SIZE, SEED


def prepare_image(image_path, label, image_size=IMAGE_SIZE, preprocess_fn=None):
    """
    Lee una imagen JPEG, la redimensiona y aplica el preprocess_input correspondiente al 
    modelo de transfer learning.

    Parámetros:
    - image_path (tf.Tensor): Ruta absoluta o relativa de la imagen (como tensor de cadena).
    - label (tf.Tensor): Etiqueta entera correspondiente a la imagen.
    - image_size (tuple): Tamaño para redimensionar la imagen (alto, ancho).
    - preprocess_fn (callable, opcional): Función de preprocesamiento específica del modelo de Keras.

    Returns:
    - image (tf.Tensor): El tensor de la imagen procesada de tipo tf.float32.
    - label (tf.Tensor): El tensor de la etiqueta.
    """
    # Leer los bytes del archivo desde la ruta
    image = tf.io.read_file(image_path)

    # Decodificar el formato JPEG a un tensor de 3 canales (RGB)
    image = tf.image.decode_jpeg(image, channels=3)

    # Redimensionar la imagen
    image = tf.image.resize(image, image_size)

    # Convertir los valores de los píxeles a punto flotante
    image = tf.cast(image, tf.float32)

    # Aplicar el preprocesamiento específico del modelo
    if preprocess_fn is not None:
        image = preprocess_fn(image)

    return image, label


def create_dataset(images_dir, csv_file, split, batch=True, batch_size=32, image_size=IMAGE_SIZE, preprocess_fn=None):
    """
    Crea un tf.data.Dataset para train, val o test.

    Parámetros:
    - images_dir (str o Path): Ruta de la carpeta en la que se encuentran las imágenes.
    - csv_file (str): Ruta del archivo CSV que contiene los datos de las imágenes. 
                      Debe contener las columnas 'filename', 'label' y 'split'.
    - split (str): Partición de los datos a cargar ('train', 'val' o 'test').
    - batch (bool): Si es True, agrupa las muestras en lotes.
    - batch_size (int): Número de muestras procesadas por lote.
    - image_size (tuple): Tamaño para redimensionar las imágenes (alto, ancho).
    - preprocess_fn (callable, opcional): Función de preprocesamiento del modelo que se pasará a `prepare_image`.

    Returns:
    - dataset (tf.data.Dataset): Conjunto de datos de TensorFlow preprocesado, agrupado en lotes.
    """
    images_dir = Path(images_dir)

    # Leer el CSV y filtrar las filas correspondientes al split
    df = pd.read_csv(csv_file)
    df_split = df[df["split"] == split].copy()

    # Lista con las rutas completas para cada imagen del dataset
    image_paths = [
        str(images_dir / filename)
        for filename in df_split["filename"]
    ]

    # Mapear las etiquetas de texto a valores enteros y guardarlas en un arreglo
    df_split["label_map"] = df_split["label"].map(LABEL_MAP)
    labels = df_split["label_map"].astype("int32").values

    # Crear el dataset inicial emparejando cada ruta de texto con su etiqueta
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # El shuffle se hace sólo para el conjunto de entrenamiento
    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=len(df_split),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    # Convertir las rutas de texto en las imágenes procesadas reales
    # Por cada ruta en el dataset, ejecuta 'prepare_image'
    dataset = dataset.map(
        lambda x, y: prepare_image(
            image_path=x,
            label=y,
            image_size=image_size,
            preprocess_fn=preprocess_fn
        ),
        num_parallel_calls=tf.data.AUTOTUNE # Hace que el proceso se ejecute en paralelo
    )

    # Agrupar los datos en lotes del tamaño especificado sólo si se solicita
    if batch:
        dataset = dataset.batch(batch_size)

    # Precargar el siguiente lote en memoria (CPU) mientras la GPU/CPU entrena el lote actual
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_dataset_splits(upp_imgs_dir=UPP_IMGS_DIR, upp_csv_file=UPP_CSV_FILE, 
                       piid_imgs_dir=PIID_IMGS_DIR, piid_csv_file=PIID_CSV_FILE, 
                       batch_size=32, image_size=IMAGE_SIZE, preprocess_fn=None):
    """
    Carga y genera los conjuntos de datos de entrenamiento, validación y prueba.

    Parámetros:
    - upp_imgs_dir (str o Path): Ruta de la carpeta que contiene las imágenes del dataset principal.
    - upp_csv_file (str): Ruta del archivo CSV que contiene los datos de las imágenes del dataset principal.
                          Debe contener las columnas 'filename', 'label' y 'split'.
    - piid_imgs_dir (str o Path): Ruta de la carpeta que contiene las imágenes de PIID.
    - piid_csv_file (str): Ruta del archivo CSV del dataset PIID.
                           Debe contener las columnas 'filename', 'label' y 'split'.
    - batch_size (int): Número de muestras procesadas por lote.
    - image_size (tuple): Tamaño para redimensionar las imágenes (alto, ancho).
    - preprocess_fn (callable, opcional): Función de preprocesamiento del modelo.

    Returns:
    - tuple: Una tupla con los tres conjuntos de datos (train_ds, val_ds, test_ds).
    """
    # Cargar el dataset principal
    train_ds = create_dataset(
        images_dir=upp_imgs_dir,
        csv_file=upp_csv_file,
        split="train",
        batch=True,
        batch_size=batch_size,
        image_size=image_size,
        preprocess_fn=preprocess_fn
    )

    val_ds = create_dataset(
        images_dir=upp_imgs_dir,
        csv_file=upp_csv_file,
        split="val",
        batch=True,
        batch_size=batch_size,
        image_size=image_size,
        preprocess_fn=preprocess_fn
    )

    test_ds = create_dataset(
        images_dir=upp_imgs_dir,
        csv_file=upp_csv_file,
        split="test",
        batch=False,
        batch_size=batch_size,
        image_size=image_size,
        preprocess_fn=preprocess_fn
    )

    # Cargar los datos de PIID y concatenarlos a test_ds
    piid_test_ds = create_dataset(
        images_dir=piid_imgs_dir,
        csv_file=piid_csv_file,
        split="test",
        batch=False,
        batch_size=batch_size,
        image_size=image_size,
        preprocess_fn=preprocess_fn
    )

    # Concatenar ambos datasets antes del batching para mantener lotes uniformes
    test_ds = (
        test_ds
        .concatenate(piid_test_ds)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


def get_class_weights(csv_file=UPP_CSV_FILE):
    """
    Calcula los pesos balanceados de las clases utilizando las etiquetas del conjunto
    de entrenamiento definido en el CSV.

    Parámetros:
    - csv_file (str o Path): Ruta del archivo CSV que contiene los datos de las imágenes 
                            del dataset. Debe contener las columnas 'label' y 'split'.

    Returns:
    - class_weights (dict): Diccionario con el peso asociado a cada clase.
    """
    # Leer el CSV
    df = pd.read_csv(csv_file)

    # Seleccionar sólo las muestras de entrenamiento
    df_train = df[df["split"] == "train"]

    # Convertir etiquetas de texto a enteros
    y_train = df_train["label"].map(LABEL_MAP).values

    # Obtener las clases únicas en el dataset
    classes = np.unique(y_train)

    # Calcular los pesos balanceados para cada clase
    weights = compute_class_weight(
        class_weight="balanced", # Asigna mayor peso a las clases menos frecuentes
        classes=classes, 
        y=y_train
    )

    # Crear diccionario {clase: peso}
    class_weights = {int(k): float(v) for k, v in zip(classes, weights)}

    return class_weights
