import tensorflow as tf
from tensorflow.keras import layers, models
from upp_classification.config import INPUT_SHAPE, NUM_CLASSES


def build_model(params, base_model_fn, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Construye y compila un modelo de clasificación con transfer learning,
    utilizando un diccionario con hiperparámetros.

    El modelo base es recibido como argumento para permitir utilizar diferentes
    arquitecturas (ConvNeXt, ResNet, DenseNet, etc.) manteniendo la misma
    estructura de clasificación.

    Args:
    - params (dict): Diccionario con los hiperparámetros seleccionados. Debe contener:
                     - dense_units (list[int]): Número de neuronas por capa densa.
                     - dropout_rate (float): Tasa de dropout aplicada antes de la capa de salida.
                     - optimizer (str): Optimizador (Adam, AdamW o SGD).
                     - learning_rate (float): Tasa de aprendizaje del optimizador.
                     - weight_decay (float): Decaimiento de pesos para AdamW.
                     - momentum (float): Momentum para SGD.
                     - nesterov (bool): Indica si se usa momentum de Nesterov en SGD.
    - base_model_fn (callable): Función constructora del modelo base de Keras.
                                (ej. tensorflow.keras.applications.ConvNeXtTiny).
    - input_shape (tuple, optional): Dimensiones del tensor de entrada. Por defecto es INPUT_SHAPE.
    - num_classes (int, optional): Número total de clases a predecir en la capa de salida. 
                                   Por defecto es NUM_CLASSES.

    Returns:
    - tensorflow.keras.Model: Modelo de Keras compilado.
    """
    # Capas de aumento de datos
    # Estas transformaciones se aplicarán en cada batch sólo durante el fit()
    data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.05),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05)
    ], name="data_augmentation")

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

    # Aplicar aumento de datos
    x = data_augmentation(inputs) 

    # Extraer características con el modelo base
    # training=False mantiene el comportamiento de BatchNormalization
    x = base_model(x, training=False) 
    
    # Convierte los mapas de características 2D en un vector 1D
    x = layers.GlobalAveragePooling2D()(x)

    # Agregar las capas densas con el número de neuronas definido en params
    for units in params["dense_units"]:
        x = layers.Dense(units=units, activation="relu")(x)

    # Regularización para evitar sobreajuste
    x = layers.Dropout(rate=params["dropout_rate"])(x)

    # Capa de salida (se obtienen las probabilidades por clase)
    outputs = layers.Dense(units=num_classes, activation="softmax")(x)

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

    elif params["optimizer"] == "SGD":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=params["learning_rate"],
            momentum=params["momentum"],
            nesterov=params["nesterov"],
        )

    # Compilar el modelo
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
