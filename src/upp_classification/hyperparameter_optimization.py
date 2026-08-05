import gc
import mlflow
import optuna
from tensorflow.keras import backend as K
from optuna_integration.tfkeras import TFKerasPruningCallback
from upp_classification.mlflow_tracking import generate_run_name, log_params_to_mlflow
from upp_classification.data_loader import get_dataset_splits, get_class_weights
from upp_classification.model_builder import build_model
from upp_classification.training import train_model
from upp_classification.evaluation import evaluate_model_datasets
from upp_classification.config import SEED, OPTUNA_DIR


def suggest_hyperparameters(trial, search_space):
    """
    Define el espacio de búsqueda de hiperparámetros y devuelve un diccionario con
    la configuración seleccionada por Optuna para un trial específico.

    Args:
    - trial (optuna.trial.Trial): Objeto de Optuna utilizado para sugerir valores de 
                                  hiperparámetros dentro del espacio de búsqueda definido.
    - search_space (dict): Diccionario que define el espacio de búsqueda de cada hiperparámetro.
                           Por ejemplo:
                           search_space = {
                               "dense_layers": {"low": 0, "high": 2},
                               "dense_units": [32, 64, 128, 256],
                               "dropout_rate": {"low": 0.2, "high": 0.5, "step": 0.1},
                               "optimizer_params": {
                                   "Adam": {
                                       "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True}
                                   },
                                   "AdamW": {
                                       "learning_rate": {"low": 1e-5, "high": 1e-3, "log": True},
                                       "weight_decay": {"low": 1e-7, "high": 1e-2, "log": True}
                                   }
                               },
                               "batch_size": [16, 32]
                           }

    Returns:
    - dict: Diccionario con los hiperparámetros seleccionados para el trial actual.
            - dense_layers (int): Número de capas densas.
            - dense_units (list[int]): Número de neuronas por capa densa.
            - dropout_rate (float): Tasa de dropout aplicada antes de la capa de salida.
            - optimizer (str): Optimizador.
            - learning_rate (float): Tasa de aprendizaje del optimizador.
            - weight_decay (float): Decaimiento de pesos para AdamW.
            - batch_size (int): Tamaño de batch.
    """
    params = {
        "dense_layers": trial.suggest_int(
            name="dense_layers", 
            low=search_space["dense_layers"]["low"],
            high=search_space["dense_layers"]["high"]
        ),
        "dropout_rate": trial.suggest_float(
            name="dropout_rate", 
            low=search_space["dropout_rate"]["low"],
            high=search_space["dropout_rate"]["high"],
            step=search_space["dropout_rate"]["step"]
        ),
        "optimizer": trial.suggest_categorical(name="optimizer", choices=list(search_space["optimizer_params"].keys())),
        "batch_size":  trial.suggest_categorical(name="batch_size", choices=search_space["batch_size"])
    }

    # Número de neuronas en cada capa densa
    params["dense_units"] = [] # Si dense_layers=0, la lista permanecerá vacía
    for i in range(params["dense_layers"]):
        # Se crea un hiperparámetro independiente por cada capa densa  (dense_units_1, dense_units_2, ...)
        units = trial.suggest_categorical(name=f"dense_units_{i+1}", choices=search_space["dense_units"])
        params["dense_units"].append(units)
    
    # Hiperparámetros para cada optimizador
    optimizer_params = search_space["optimizer_params"][params["optimizer"]]

    if params["optimizer"] == "Adam":
        lr = optimizer_params["learning_rate"]
        params["learning_rate"] = trial.suggest_float(name="lr_adam", low=lr["low"], high=lr["high"], log=lr["log"])
    
    elif params["optimizer"] == "AdamW":
        lr = optimizer_params["learning_rate"]
        params["learning_rate"] = trial.suggest_float(name="lr_adamw", low=lr["low"], high=lr["high"], log=lr["log"])
        
        wd = optimizer_params["weight_decay"]
        params["weight_decay"] = trial.suggest_float(name="weight_decay", low=wd["low"], high=wd["high"], log=wd["log"])

    return params


def objective(trial, base_model_fn, preprocess_fn, search_space, image_size, class_weights, use_cache=False):
    """
    Función objetivo utilizada por Optuna para evaluar una configuración concreta de hiperparámetros.

    Args:
    - trial (optuna.trial.Trial): Objeto de Optuna que gestiona la selección de hiperparámetros
                                  para la prueba actual.
    - base_model_fn (callable): Función constructora del modelo base utilizado en el entrenamiento.
                                (ej. tensorflow.keras.applications.ConvNeXtTiny).
    - preprocess_fn (callable): Función de preprocesamiento asociada al modelo base.
                                (ej. convnext.preprocess_input).
    - search_space (dict): Diccionario que define el espacio de búsqueda de cada hiperparámetro.
    - image_size (tuple): Tamaño para redimensionar las imágenes (alto, ancho).
    - class_weights (dict): Diccionario con los pesos balanceados de cada clase, calculados a partir 
                            del conjunto de entrenamiento.
    - use_cache (bool, optional): Si es True, habilita el uso de caché en disco para acelerar
                                  la carga y procesamiento de los datos. Por defecto es False.

    Returns:
    - float: Métrica objetivo que Optuna intenta minimizar (en este caso es val_loss).

    Raises:
    - optuna.TrialPruned: Si el callback de pruning determina que el trial no es prometedor 
                          y detiene el entrenamiento de forma anticipada.
    - Exception: Cualquier excepción producida durante la carga de los datos, construcción 
                 del modelo, entrenamiento o evaluación.
    """
    # Obtener hiperparámetros sugeridos por Optuna
    params = suggest_hyperparameters(trial=trial, search_space=search_space)

    # Generar el nombre del run de MLflow
    model_name = base_model_fn.__name__
    run_name = generate_run_name(architecture_name=model_name, params=params)

    # Crear un run de MLflow asociado al trial actual de Optuna
    with mlflow.start_run(run_name=run_name):
        # Registrar información adicional del run
        mlflow.set_tag("architecture", model_name)
        mlflow.set_tag("optuna_trial", trial.number)
        mlflow.set_tag("stage", "hyperparameter_optimization")

        # Registrar los hiperparámetros en MLflow
        log_params_to_mlflow(params=params)

        # Callback para detener trials poco prometedores durante la optimización
        pruning_callback = TFKerasPruningCallback(
            trial=trial,
            monitor="val_loss"
        )

        try:
            # Cargar los datasets con el batch_size seleccionado por Optuna
            train_ds, val_ds = get_dataset_splits(
                image_size=image_size,
                batch_size=params["batch_size"],
                use_cache=use_cache,
                include_test=False
            )
            
            # Construir el modelo con los hiperparámetros seleccionados
            model = build_model(
                params=params, 
                base_model_fn=base_model_fn,
                preprocess_fn=preprocess_fn,
                input_shape=(*image_size, 3),
                use_augmentation=True
            )

            # Entrenar el modelo con la configuración actual
            history = train_model(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                class_weights=class_weights,
                use_mlflow=True,
                pruning_callback=pruning_callback
            )

            # Evaluar el modelo en los conjuntos de entrenamiento y validación
            # y registrar las métricas en MLflow
            evaluate_model_datasets(
                model=model, 
                train_ds=train_ds, 
                val_ds=val_ds, 
                return_predictions=False,
                include_class_metrics=False,
                use_mlflow=True
            )

            # Obtener la menor pérdida de validación alcanzada durante el entrenamiento
            val_loss = min(history.history["val_loss"])
            
            # Registrar métrica objetivo de Optuna en MLflow
            mlflow.log_metric(key="best_val_loss", value=val_loss)

            # Marcar el run de MLflow como completado correctamente
            mlflow.set_tag("status", "completed")
        
            return val_loss

        except optuna.TrialPruned:
            # Marcar el run de MLflow como trial descartado por pruning
            mlflow.set_tag("status", "pruned")
            raise

        except Exception as e:
            # Marcar el run de MLflow como trial finalizado con error
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(e))
            raise

        finally:
            # Eliminar la referencia al modelo entrenado
            if 'model' in locals():
                del model

            # Eliminar los pipelines de datos para vaciar el búfer de prefetch
            if 'train_ds' in locals():
                del train_ds
            if 'val_ds' in locals():
                del val_ds

            # Forzar la liberación de memoria RAM/VRAM al terminar el trial
            K.clear_session()
            gc.collect()


def run_hyperparameter_search(base_model_fn, preprocess_fn, search_space, image_size, n_trials=60, optuna_dir=OPTUNA_DIR, use_cache=False):
    """
    Ejecuta la búsqueda de hiperparámetros utilizando Optuna.

    Args:
    - base_model_fn (callable): Función constructora del modelo base utilizado en el entrenamiento.
                                (ej. tensorflow.keras.applications.ConvNeXtTiny).
    - preprocess_fn (callable): Función de preprocesamiento asociada al modelo base.
                                (ej. convnext.preprocess_input).
    - search_space (dict): Diccionario que define el espacio de búsqueda de cada hiperparámetro.
    - image_size (tuple): Tamaño para redimensionar las imágenes (alto, ancho).
    - n_trials (int, optional): Número total de configuraciones de hiperparámetros que se desea
                                evaluar. Si el estudio ya existe, únicamente se ejecutarán los
                                trials faltantes para alcanzar este número. Por defecto es 60.
    - optuna_dir (str o Path, optional): Directorio donde se almacenará la base de datos SQLite del 
                                         estudio de Optuna. Por defecto es OPTUNA_DIR.
    - use_cache (bool, optional): Si es True, habilita el uso de caché en disco para acelerar
                                  la carga y procesamiento de los datos. Por defecto es False.
    
    Returns:
    - optuna.study.Study: Objeto Study de Optuna con los resultados completos de la optimización.
                          Permite acceder a:
                          - best_value: mejor métrica obtenida.
                          - best_params: mejores hiperparámetros encontrados.
                          - best_trial: mejor trial.
                          - trials: historial completo de pruebas.
    """
    # Nombre de la arquitectura utilizada
    model_name = base_model_fn.__name__

    # Obtener los pesos balanceados de las clases
    class_weights = get_class_weights()

    # Configurar pruner para detener trials con bajo rendimiento
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8, # Trials iniciales que se ejecutan completos antes de comenzar a aplicar pruning
        n_warmup_steps=7, # Número de épocas iniciales de cada trial durante las que no se evalúa pruning
        interval_steps=1, # Frecuencia (en épocas) con la que se revisa si un trial debe ser detenido
        n_min_trials=3 # Número mínimo de trials que deben alcanzar una época para poder evaluar el pruning
    )

    # Crear el estudio de Optuna
    study = optuna.create_study(
        direction="minimize",
        study_name=f"optimization_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{optuna_dir / f'{model_name}.db'}",
        pruner=pruner,
        load_if_exists=True
    )

    # Número de trials registrados en el estudio como completados o detenidos por pruning
    completed_trials = len(
        study.get_trials(
            states=(
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED
            )
        )
    )
    
    # Ejecutar la optimización de hiperparámetros
    study.optimize(
        lambda trial:
            objective(
                trial=trial, 
                base_model_fn=base_model_fn, 
                preprocess_fn=preprocess_fn,
                search_space=search_space,
                image_size=image_size,
                class_weights=class_weights,
                use_cache=use_cache
            ),
        n_trials=max(0, n_trials - completed_trials) # Ejecutar únicamente los trials faltantes para alcanzar n_trials
    )

    return study


def reconstruct_best_params(best_params):
    """
    Convierte el diccionario plano devuelto por Optuna en la estructura anidada
    que espera la función build_model().

    Args:
    - best_params (dict): Diccionario con los hiperparámetros devueltos 
                          por study.best_params de Optuna.

    Returns:
    - dict: Diccionario jerárquico con el fomarto que espera build_model().
    """
    params = {
        "dense_layers": best_params["dense_layers"],
        "dropout_rate": best_params["dropout_rate"],
        "optimizer": best_params["optimizer"],
        "batch_size": best_params["batch_size"],
        "dense_units": []
    }
    
    # Recuperar las neuronas por cada capa densa configurada
    for i in range(params["dense_layers"]):
        params["dense_units"].append(best_params[f"dense_units_{i+1}"])
        
    # Recuperar los parámetros específicos del optimizador
    if params["optimizer"] == "Adam":
        params["learning_rate"] = best_params["lr_adam"]
        
    elif params["optimizer"] == "AdamW":
        params["learning_rate"] = best_params["lr_adamw"]
        params["weight_decay"] = best_params["weight_decay"]
        
    return params
