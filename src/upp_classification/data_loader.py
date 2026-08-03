import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from upp_classification.config import UPP_IMGS_DIR, UPP_CSV_FILE, PIID_IMGS_DIR, PIID_CSV_FILE, LABEL_MAP, CACHE_DIR, SEED


def prepare_image(image_path, image_size, label=None):
    """
    Lee una imagen JPEG, la redimensiona y convierte sus valores a punto flotante.

    Args:
    - image_path (str o tf.Tensor): Ruta absoluta o relativa de la imagen (como cadena de texto o tensor).
    - image_size (tuple): Tamaño para redimensionar la imagen (alto, ancho).
    - label (int o tf.Tensor, optional): Etiqueta entera correspondiente a la imagen. Por defecto es None.

    Returns:
    - tf.Tensor o tuple: 
        - Si 'label' es None: Devuelve el tensor de la imagen procesada de tipo tf.float32.
        - Si se proporciona 'label': Devuelve una tupla (image, label) con sus respectivos tensores.
    """
    # Leer los bytes del archivo desde la ruta
    image = tf.io.read_file(image_path)

    # Decodificar el formato JPEG a un tensor de 3 canales (RGB)
    image = tf.image.decode_jpeg(image, channels=3)

    # Redimensionar la imagen
    image = tf.image.resize(image, image_size)

    # Convertir los valores de los píxeles a punto flotante
    image = tf.cast(image, tf.float32)

    # Si se proporcionó una etiqueta, devuelve la tupla
    if label is not None:
        return image, label

    # Si no hay etiqueta, sólo devuelve la imagen
    return image


def create_dataset(images_dir, csv_file, image_size, split, batch=True, batch_size=32, use_cache=False, cache_name=None):
    """
    Crea un tf.data.Dataset para train, val o test.

    Args:
    - images_dir (str o Path): Ruta de la carpeta en la que se encuentran las imágenes.
    - csv_file (str): Ruta del archivo CSV que contiene los metadatos de las imágenes. 
                      Debe contener las columnas 'filename', 'label' y 'split'.
    - image_size (tuple): Tamaño para redimensionar las imágenes (alto, ancho).
    - split (str): Partición de los datos a cargar ('train', 'val' o 'test').
    - batch (bool, optional): Si es True, agrupa las muestras en lotes. Por defecto es True.
    - batch_size (int, optional): Número de muestras procesadas por lote. Por defecto es 32.
    - use_cache (bool, optional): Si es True, almacena las imágenes preprocesadas en disco 
                                  para acelerar el entrenamiento en épocas posteriores. 
                                  Por defecto es False.
    - cache_name (str, optional): Nombre o prefijo del archivo de caché en disco. Debe
                                  proporcionarse si use_cache es True. Por defecto es None.

    Returns:
    - tf.data.Dataset: Conjunto de datos de TensorFlow preprocesado, agrupado en lotes.
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

    # Convertir las rutas de texto en las imágenes procesadas reales
    # Por cada ruta en el dataset, ejecuta 'prepare_image'
    dataset = dataset.map(
        lambda x, y: prepare_image(
            image_path=x,
            label=y,
            image_size=image_size
        ),
        num_parallel_calls=tf.data.AUTOTUNE # Hace que el proceso se ejecute en paralelo
    )

    # Almacenar el dataset en caché (en disco) si se solicita, para evitar
    # repetir la lectura y redimensionamiento de las imágenes en cada época
    if use_cache and cache_name:
        dataset = dataset.cache(str(CACHE_DIR / cache_name))

    # El shuffle se hace sólo para el conjunto de entrenamiento
    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=len(df_split),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    # Agrupar los datos en lotes del tamaño especificado sólo si se solicita
    if batch:
        dataset = dataset.batch(batch_size)

    # Precargar el siguiente lote en memoria (CPU) mientras la GPU/CPU entrena el lote actual
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_dataset_splits(image_size, upp_imgs_dir=UPP_IMGS_DIR, upp_csv_file=UPP_CSV_FILE, 
                       piid_imgs_dir=PIID_IMGS_DIR, piid_csv_file=PIID_CSV_FILE, 
                       batch_size=32, use_cache=False):
    """
    Carga y genera los conjuntos de datos de entrenamiento, validación y prueba.

    Args:
    - image_size (tuple): Tamaño para redimensionar las imágenes (alto, ancho).
    - upp_imgs_dir (str o Path, optional): Ruta de la carpeta que contiene las imágenes del dataset principal.
                                           Por defecto es UPP_IMGS_DIR.
    - upp_csv_file (str, optional): Ruta del archivo CSV que contiene los metadatos de las imágenes del dataset
                                    principal. Debe contener las columnas 'filename', 'label' y 'split'.
                                    Por defecto es UPP_CSV_FILE.
    - piid_imgs_dir (str o Path, optional): Ruta de la carpeta que contiene las imágenes de PIID.
                                            Por defecto es PIID_IMGS_DIR.
    - piid_csv_file (str, optional): Ruta del archivo CSV que contiene los metadatos de las imágenes del dataset 
                                     PIID. Debe contener las columnas 'filename', 'label' y 'split'. 
                                     Por defecto es PIID_CSV_FILE.
    - batch_size (int, optional): Número de muestras procesadas por lote. Por defecto es 32.
    - use_cache (bool, optional): Si es True, habilita el guardado en caché en disco para acelerar la carga y
                                  procesamiento de los datos. Por defecto es False.

    Returns:
    - tuple: Una tupla que contiene:
             - train_ds (tf.data.Dataset): Conjunto de datos de entrenamiento.
             - val_ds (tf.data.Dataset): Conjunto de datos de validación.
             - test_ds (tf.data.Dataset): Conjunto de datos de prueba.
    """
    # Cargar el dataset principal
    train_ds = create_dataset(
        images_dir=upp_imgs_dir,
        csv_file=upp_csv_file,
        image_size=image_size,
        split="train",
        batch=True,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_name=f"upp_train_{image_size[0]}x{image_size[1]}"
    )

    val_ds = create_dataset(
        images_dir=upp_imgs_dir,
        csv_file=upp_csv_file,
        image_size=image_size,
        split="val",
        batch=True,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_name=f"upp_val_{image_size[0]}x{image_size[1]}"
    )

    test_ds = create_dataset(
        images_dir=upp_imgs_dir,
        csv_file=upp_csv_file,
        image_size=image_size,
        split="test",
        batch=False,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_name=f"upp_test_{image_size[0]}x{image_size[1]}"
    )

    # Cargar los datos de PIID y concatenarlos a test_ds
    piid_test_ds = create_dataset(
        images_dir=piid_imgs_dir,
        csv_file=piid_csv_file,
        image_size=image_size,
        split="test",
        batch=False,
        batch_size=batch_size,
        use_cache=use_cache,
        cache_name=f"piid_test_{image_size[0]}x{image_size[1]}"
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

    Args:
    - csv_file (str o Path, optional): Ruta del archivo CSV que contiene los metadatos de las imágenes 
                                       del dataset. Debe contener las columnas 'label' y 'split'.
                                       Por defecto es UPP_CSV_FILE.

    Returns:
    - dict: Diccionario con el peso asociado a cada clase.
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
