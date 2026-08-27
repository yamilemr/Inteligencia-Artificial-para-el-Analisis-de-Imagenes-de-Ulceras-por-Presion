"""
Entrena y evalúa el mejor modelo obtenido de la optimización con Optuna, registra las métricas 
y guarda el modelo final utilizando MLflow.

Comandos de ejecución (desde la raíz del proyecto):

- Ejecutar el entrenamiento del mejor modelo:
    uv run python scripts/train_best_model.py --model ResNet50V2 --experiment upp_classification --cache --augmentation

    Nota: 
    - En caso de que no se desee usar caché en disco para cargar las imágenes, se omite el parámetro --cache en el comando de ejecución.
    - En caso de que no se desee usar aumento de datos durante el entrenamiento, se omite el parámetro --augmentation en el comando de ejecución.

- Abrir la interfaz de MLflow para visualizar los resultados:
    mlflow ui --backend-store-uri sqlite:///experiments/mlflow_tracking.db
"""

import argparse
import optuna
import mlflow

from tensorflow.keras.applications import ResNet50V2, InceptionResNetV2, DenseNet121, ConvNeXtTiny
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet50v2_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet121_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnexttiny_preprocess

from upp_classification.data_loader import get_dataset_splits, get_class_weights
from upp_classification.mlflow_tracking import setup_mlflow, log_params_to_mlflow, log_and_save_model
from upp_classification.model_builder import build_model
from upp_classification.training import train_model
from upp_classification.hyperparameter_optimization import reconstruct_best_params
from upp_classification.evaluation import evaluate_model_datasets
from upp_classification.config import AVAILABLE_MODELS, OPTUNA_DIR


# Diccionario que relaciona el nombre de la arquitectura con su función constructora
# y la función de preprocesamiento correspondiente
MODELS = {
    "ResNet50V2": (ResNet50V2, resnet50v2_preprocess),
    "InceptionResNetV2": (InceptionResNetV2, inceptionresnetv2_preprocess),
    "DenseNet121": (DenseNet121, densenet121_preprocess),
    "ConvNeXtTiny": (ConvNeXtTiny, convnexttiny_preprocess),
    "CustomCNN": (None, None)
}


def parse_args():
    """
    Procesa los argumentos proporcionados desde la línea de comandos.

    Returns:
    - argparse.Namespace: Objeto que contiene los argumentos introducidos por el usuario:
                          - model (str): Arquitectura de transfer learning.
                          - experiment (str): Nombre del experimento de MLflow.
                          - cache (bool): Habilita el uso de caché en disco para acelerar 
                                          la carga de imágenes.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model",
        required=True,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Arquitectura de transfer learning."
    )
    
    parser.add_argument(
        "--experiment",
        default="upp_classification",
        help="Nombre del experimento de MLflow."
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Habilita el uso de caché en disco para acelerar la carga de imágenes."
    )

    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Habilita el aumento de datos durante el entrenamiento."
    )
    
    return parser.parse_args()


def main():
    """
    Entrena y evalúa el mejor modelo encontrado por Optuna para la arquitectura seleccionada.

    La función procesa los argumentos de entrada, recupera los mejores hiperparámetros desde
    la base de datos de Optuna, ejecuta el entrenamiento del modelo, lo evalúa en todos los
    conjuntos de datos y finalmente lo registra en MLflow y lo guarda de forma local.

    Returns:
    - None
    """
    # Leer los argumentos de la línea de comandos
    args = parse_args()

    # Nombre del modelo
    model_name = args.model

    # Tamaño para redimensionar las imágenes según el modelo
    image_size = AVAILABLE_MODELS[model_name]["image_size"]

    # Cargar el estudio de Optuna
    study_name = f"optimization_{model_name}"
    storage_uri = f"sqlite:///{OPTUNA_DIR / f'{model_name}.db'}"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
    except KeyError:
        print(f"Error: No se encontró un estudio de Optuna para {model_name}.")
        return
    
    # Reconstruir los mejores hiperparámetros obtenidos por Optuna
    best_params = reconstruct_best_params(study.best_params)
        
    # Configurar el experimento de MLflow
    setup_mlflow(experiment_name=args.experiment)

    # Crear un run de MLflow para el mejor modelo
    run_name = f"BEST MODEL | {model_name}"
    
    with mlflow.start_run(run_name=run_name):
        # Registrar información adicional del run
        mlflow.set_tag("architecture", model_name)
        mlflow.set_tag("stage", "best_model_training")
        
        # Registrar los hiperparámetros en MLflow
        log_params_to_mlflow(params=best_params)
        
        # Cargar los datasets y los pesos balanceados de las clases
        train_ds, val_ds, test_ds = get_dataset_splits(
            image_size=image_size, 
            batch_size=best_params["batch_size"],
            use_cache=args.cache,
            include_test=True
        )
        class_weights = get_class_weights()
        
        # Construir el modelo
        base_model_fn, preprocess_fn = MODELS[model_name]
        model = build_model(
            params=best_params,
            base_model_fn=base_model_fn,
            preprocess_fn=preprocess_fn,
            input_shape=(*image_size, 3),
            use_augmentation=args.augmentation
        )
        
        # Entrenar el modelo
        history = train_model(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            use_mlflow=True,
            pruning_callback=None
        )

        # Registrar en MLflow el número de épocas ejecutadas
        mlflow.log_param("epochs", len(history.epoch))
        
        # Evaluar el modelo en los conjuntos de entrenamiento, validación y prueba
        # y registrar las métricas en MLflow
        results = evaluate_model_datasets(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            return_predictions=False,
            include_class_metrics=True,
            use_mlflow=True
        )
        
        # Registrar el modelo en MLflow y guardarlo de forma local
        log_and_save_model(model=model, architecture_name=model_name)

    # Imprimir los resultados
    print(f"\nArquitectura: {model_name}")

    for ds_name, ds_results in results.items():
        metrics = ds_results["metrics"]

        print(f"\n----- {ds_name.upper()} -----")
        print(f"  Accuracy = {metrics['accuracy']:.4f}")
        print(f"  Precision = {metrics['precision']:.4f}")
        print(f"  Recall = {metrics['recall']:.4f}")
        print(f"  F1-score  = {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
