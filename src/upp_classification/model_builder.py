import tensorflow as tf
from tensorflow.keras import layers, models
from upp_classification.config import NUM_CLASSES


def build_model(params, base_model_fn, input_shape, preprocess_fn=None, num_classes=NUM_CLASSES, use_augmentation=True):
    """
    Construye y compila un modelo de clasificación con transfer learning, utilizando un 
    diccionario con hiperparámetros.

    El modelo base es recibido como argumento para permitir utilizar diferentes arquitecturas
    (ConvNeXt, ResNet, DenseNet, etc.) manteniendo la misma estructura de clasificación.

    Args:
    - params (dict): Diccionario con los hiperparámetros seleccionados. Debe contener:
                     - dense_units (list[int]): Número de neuronas por capa densa.
                     - dropout_rate (float): Tasa de dropout aplicada después de cada capa densa
                                             o directamente después del GlobalAveragePooling2D si
                                             no existen capas densas.
                     - optimizer (str): Optimizador (Adam o AdamW).
                     - learning_rate (float): Tasa de aprendizaje inicial del optimizador.
                     - weight_decay (float): Decaimiento de pesos (sólo para AdamW).
    - base_model_fn (callable): Función constructora del modelo base de Keras.
                                (ej. tensorflow.keras.applications.ConvNeXtTiny).
    - input_shape (tuple): Dimensiones del tensor de entrada (alto, ancho, canales).
    - preprocess_fn (callable, optional): Función de preprocesamiento específica del modelo de Keras. 
                                          Por defecto es None.
    - num_classes (int, optional): Número total de clases a predecir en la capa de salida. 
                                   Por defecto es NUM_CLASSES.
    - use_augmentation (bool, optional): Indica si se aplican capas de aumento de datos durante el 
                                         entrenamiento. Por defecto es True.

    Returns:
    - tensorflow.keras.Model: Modelo de Keras compilado.

    Raises:
    - ValueError: Si el optimizador especificado en params no es compatible.
    """
    # Cargar el modelo base con los pesos de ImageNet
    base_model = base_model_fn(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Congelar los pesos del modelo base
    base_model.trainable = False

    # Entrada del modelo
    inputs = layers.Input(shape=input_shape)

    # Iniciar flujo de tensores
    x = inputs

    # Aplicar aumento de datos si está indicado
    if use_augmentation:
        # Capas de aumento de datos
        # Estas transformaciones se aplicarán en cada batch sólo durante el fit()
        data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.05),
                layers.RandomTranslation(height_factor=0.05, width_factor=0.05)
        ], name="data_augmentation")

        x = data_augmentation(x) 

    # Aplicar el preprocesamiento específico del modelo
    if preprocess_fn is not None:
        x = preprocess_fn(x)

    # Extraer características con el modelo base
    # training=False mantiene el comportamiento de BatchNormalization
    x = base_model(x, training=False) 
    
    # Convierte los mapas de características 2D en un vector 1D
    x = layers.GlobalAveragePooling2D()(x)

    # Agregar las capas densas con el número de neuronas definido en params
    # Después de cada capa densa se aplica Dropout como regularización
    if params["dense_units"]:
        for units in params["dense_units"]:
            x = layers.Dense(units=units, activation="relu")(x)
            x = layers.Dropout(rate=params["dropout_rate"])(x)

    # Si no se agregan capas densas, aplicar Dropout sobre las características
    # extraídas por el modelo base
    else:
        x = layers.Dropout(rate=params["dropout_rate"])(x)

    # Capa de salida (se obtienen los logits por clase)
    outputs = layers.Dense(units=num_classes, activation=None)(x)

    # Construir el modelo final
    model = models.Model(inputs=inputs, outputs=outputs)

    # Configuración del optimizador con sus hiperparámetros correspondientes
    if params["optimizer"] == "Adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=params["learning_rate"]
        )

    elif params["optimizer"] == "AdamW":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=params["learning_rate"], 
            weight_decay=params["weight_decay"]
        )

    else:
        raise ValueError(f"Optimizador no soportado: {params['optimizer']}")

    # Compilar el modelo utilizando logits como salida de la red
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model
