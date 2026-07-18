import optuna
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from optuna_integration.tfkeras import TFKerasPruningCallback
from upp_classification.data_loader import get_dataset_splits
from upp_classification.config import INPUT_SHAPE, NUM_CLASSES, EPOCHS, SEED, OPTUNA_DIR


def suggest_hyperparameters(trial):
    """
    Define el espacio de búsqueda de hiperparámetros y devuelve un diccionario con
    la configuración seleccionada por Optuna para un trial específico.

    Parámetros:
    - trial (optuna.trial.Trial): Objeto de Optuna utilizado para sugerir valores de 
                                  hiperparámetros dentro del espacio de búsqueda definido.

    Returns:
    - params (dict): Diccionario con los hiperparámetros seleccionados para el trial actual.
                     - dense_layers (int): Número de capas densas.
                     - dense_units (list[int]): Número de neuronas por capa densa.
                     - dropout_rate (float): Tasa de dropout aplicada antes de la capa de salida.
                     - optimizer (str): Optimizador.
                     - batch_size (int): Tamaño de batch.
                     - learning_rate (float): Tasa de aprendizaje del optimizador.
                     - weight_decay (float): Decaimiento de pesos para AdamW.
                     - momentum (float): Momentum para SGD.
                     - nesterov (bool): Indica si se usa momentum de Nesterov en SGD.
    """
    params = {
        "dense_layers": trial.suggest_int(name="dense_layers", low=0, high=2),
        "dropout_rate": trial.suggest_float(name="dropout_rate", low=0.2, high=0.5, step=0.1),
        "optimizer": trial.suggest_categorical(name="optimizer", choices=["Adam", "AdamW", "SGD"]),
        "batch_size": trial.suggest_categorical(name="batch_size", choices=[16, 32])
    }

    # Número de neuronas en cada capa densa
    params["dense_units"] = [] # Si dense_layers=0, la lista permanecerá vacía
    for i in range(params["dense_layers"]):
        # Se crea un hiperparámetro independiente por cada capa densa  (dense_units_1, dense_units_2)
        units = trial.suggest_categorical(name=f"dense_units_{i+1}", choices=[32, 64, 128])
        params["dense_units"].append(units)
        
    # Hiperparámetros para cada optimizador
    if params["optimizer"] == "Adam":
        params["learning_rate"] = trial.suggest_float(name="lr_adam", low=1e-4, high=1e-2, log=True)
    
    elif params["optimizer"] == "AdamW":
        params["learning_rate"] = trial.suggest_float(name="lr_adamw", low=1e-4, high=1e-2, log=True)
        params["weight_decay"] = trial.suggest_float(name="weight_decay", low=1e-6, high=1e-2, log=True)
    
    elif params["optimizer"] == "SGD":
        params["learning_rate"] = trial.suggest_float(name="lr_sgd", low=1e-3, high=1e-1, log=True)
        params["momentum"] = trial.suggest_float(name="momentum", low=0.8, high=0.99)
        params["nesterov"] = trial.suggest_categorical(name="nesterov", choices=[True, False])

    return params


def build_model(params, base_model_fn):
    """
    Construye y compila un modelo de clasificación con transfer learning, 
    utilizando los hiperparámetros seleccionados por Optuna.

    El modelo base es recibido como argumento para permitir utilizar diferentes
    arquitecturas (ConvNeXt, ResNet, DenseNet, etc.) manteniendo la misma
    estructura de clasificación.

    Parámetros:
    - params (dict): Diccionario con los hiperparámetros seleccionados por Optuna.
                     Debe contener:
                     - dense_layers (int): Número de capas densas.
                     - dense_units (list[int]): Número de neuronas por capa densa.
                     - dropout_rate (float): Tasa de dropout aplicada antes de la capa de salida.
                     - optimizer (str): Optimizador.
                     - batch_size (int): Tamaño de batch.
                     - learning_rate (float): Tasa de aprendizaje del optimizador.
                     - weight_decay (float): Decaimiento de pesos para AdamW.
                     - momentum (float): Momentum para SGD.
                     - nesterov (bool): Indica si se usa momentum de Nesterov en SGD.
    - base_model_fn (callable): Función constructora del modelo base de Keras.
                                (ej. tensorflow.keras.applications.ConvNeXtTiny)

    Returns:
    - model (tensorflow.keras.Model): Modelo compilado listo para entrenamiento.
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
        input_shape=INPUT_SHAPE
    )

    # Congelar los pesos del modelo base
    base_model.trainable = False

    # Entrada del modelo
    inputs = layers.Input(shape=INPUT_SHAPE)

    # Aplicar aumento de datos
    x = data_augmentation(inputs) 

    # Extraer características con el modelo base
    # training=False mantiene el comportamiento de BatchNormalization
    x = base_model(x, training=False) 
    
    # Convierte los mapas de características 2D en un vector 1D
    x = layers.GlobalAveragePooling2D()(x)

    # Agregar las capas densas con el número de neuronas seleccionado por Optuna
    for units in params["dense_units"]:
        x = layers.Dense(units=units, activation="relu")(x)

    # Regularización para evitar sobreajuste
    x = layers.Dropout(rate=params["dropout_rate"])(x)

    # Capa de salida (se obtienen las probabilidades por clase)
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(x)

    # Construir el modelo final
    model = models.Model(inputs=inputs, outputs=outputs)

    # Configuración del optimizador
    if params["optimizer"] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])

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


def train_and_evaluate(trial, params, base_model_fn, preprocess_fn):
    """
    Carga los datos, construye el modelo, realiza el entrenamiento y devuelve
    la mejor pérdida de validación obtenida.

    Parámetros:
    - trial (optuna.trial.Trial): Objeto de Optuna que gestiona la selección de hiperparámetros
                                  para la prueba actual.
    - params (dict): Diccionario con los hiperparámetros seleccionados por Optuna.
    - base_model_fn (callable): Función constructora del modelo base de Keras.
                                (ej. tensorflow.keras.applications.ConvNeXtTiny)
    - preprocess_fn (callable): Función de preprocesamiento asociada al modelo base.
                                (ej. convnext.preprocess_input)

    Returns:
    - best_val_loss (float): Menor valor de val_loss obtenido durante el entrenamiento.
    """
    # Cargar los datasets con el batch_size seleccionado por Optuna
    train_ds, val_ds, _ = get_dataset_splits(
        batch_size=params["batch_size"],
        preprocess_fn=preprocess_fn
    )
    
    # Construir modelo con los hiperparámetros seleccionados
    model = build_model(params=params, base_model_fn=base_model_fn)
    
    # Callback para detener el entrenamiento cuando el modelo deje de mejorar
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10, # Espera 10 épocas sin mejoras antes de detenerse
        restore_best_weights=True # Restaura los pesos de la época con el mejor val_loss
    )

    # Callback para reducir la tasa de aprendizaje cuando el modelo se estanque
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5, # Reduce el learning rate a la mitad cuando no hay mejora
        patience=4, # Espera 4 épocas sin mejoras antes de reducir
        min_lr=1e-7 # Learning rate mínimo permitida
    )

    # Callback para que Optuna detenga anticipadamente los trials poco prometedores
    pruning_callback = TFKerasPruningCallback(
        trial=trial,
        monitor="val_loss"
    )
    
    # Entrenar el modelo
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        shuffle=False, # El shuffle ya se realiza al cargar los datos con tf.data.Dataset
        callbacks=[early_stopping, reduce_lr, pruning_callback]
    )
    
    # Retornar el menor error de validación
    best_val_loss = min(history.history["val_loss"])

    return best_val_loss


def objective(trial, base_model_fn, preprocess_fn):
    """
    Función objetivo utilizada por Optuna para evaluar una configuración
    concreta de hiperparámetros.

    Parámetros:
    - trial (optuna.trial.Trial): Objeto de Optuna que gestiona la selección de hiperparámetros
                                  para la prueba actual.
    - base_model_fn (callable): Función constructora del modelo base utilizado en el entrenamiento.
                                (ej. tensorflow.keras.applications.ConvNeXtTiny)
    - preprocess_fn (callable): Función de preprocesamiento asociada al modelo base.
                                (ej. convnext.preprocess_input)

    Returns:
    - val_loss (float): Métrica objetivo que Optuna intenta minimizar.
    """
    # Obtener hiperparámetros sugeridos por Optuna
    params = suggest_hyperparameters(trial=trial)

    # Entrenar y evaluar la configuración actual
    val_loss = train_and_evaluate(trial=trial, params=params, base_model_fn=base_model_fn, preprocess_fn=preprocess_fn)
    
    return val_loss


def run_hyperparameter_search(base_model_fn, preprocess_fn, model_name="model", n_trials=20):
    """
    Ejecuta la búsqueda de hiperparámetros utilizando Optuna.

    Parámetros:
    - base_model_fn (callable): Función constructora del modelo base utilizado en el entrenamiento.
                                (ej. tensorflow.keras.applications.ConvNeXtTiny)
    - preprocess_fn (callable): Función de preprocesamiento asociada al modelo base.
                                (ej. convnext.preprocess_input)
    - model_name (str): Nombre del modelo utilizado. Se emplea para nombrar el estudio de Optuna.
    - n_trials (int): Número de configuraciones de hiperparámetros que serán evaluadas.

    Returns:
    - study (optuna.study.Study): Objeto Study de Optuna con los resultados completos de la optimización.
                                  Permite acceder a:
                                  - best_value: mejor métrica obtenida.
                                  - best_params: mejores hiperparámetros encontrados.
                                  - best_trial: mejor trial.
                                  - trials: historial completo de pruebas.
    """
    # Crear el estudio de Optuna
    study = optuna.create_study(
        direction="minimize",
        study_name=f"optimization_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{OPTUNA_DIR / f'{model_name}.db'}",
        load_if_exists=True
    )

    # Ejecutar la búsqueda de hiperparámetros
    study.optimize(
        lambda trial:
            objective(trial=trial, base_model_fn=base_model_fn, preprocess_fn=preprocess_fn),
        n_trials=n_trials
    )

    return study
