"""
Realiza la optimización de hiperparámetros mediante Optuna y registra los experimentos utilizando MLflow.

Comandos de ejecución (desde la raíz del proyecto):

- Ejecutar la optimización de hiperparámetros:
    uv run python scripts/run_hyperparameter_optimization.py --model ResNet50V2 --trials 60 --experiment upp_classification

- Abrir la interfaz de MLflow para visualizar los experimentos registrados:
    mlflow ui --backend-store-uri sqlite:///experiments/mlflow_tracking.db
"""

import argparse
import json

from tensorflow.keras.applications import ResNet50V2, InceptionResNetV2, DenseNet121, ConvNeXtTiny
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet50v2_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnetv2_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet121_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnexttiny_preprocess

from upp_classification.mlflow_tracking import setup_mlflow
from upp_classification.hyperparameter_optimization import run_hyperparameter_search
from upp_classification.config import OPTUNA_DIR


# Diccionario que relaciona el nombre de la arquitectura con su función constructora
# y la función de preprocesamiento correspondiente
MODELS = {
    "ResNet50V2": (ResNet50V2, resnet50v2_preprocess),
    "InceptionResNetV2": (InceptionResNetV2, inceptionresnetv2_preprocess),
    "DenseNet121": (DenseNet121, densenet121_preprocess),
    "ConvNeXtTiny": (ConvNeXtTiny, convnexttiny_preprocess)
}


def parse_args():
    """
    Procesa los argumentos proporcionados desde la línea de comandos.

    Returns:
    - argparse.Namespace: Objeto que contiene los argumentos introducidos por el usuario:
                          - model (str): Arquitectura de transfer learning.
                          - trials (int): Número de trials para la optimización.
                          - experiment (str): Nombre del experimento de MLflow.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=MODELS.keys(),
        help="Arquitectura de transfer learning."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Número de trials."
    )

    parser.add_argument(
        "--experiment",
        default="upp_classification",
        help="Nombre del experimento de MLflow."
    )

    return parser.parse_args()


def save_best_trial(study, architecture_name):
    """
    Guarda en formato JSON la información correspondiente al mejor trial obtenido durante la búsqueda de hiperparámetros.

    Args:
    - study (optuna.study.Study): Estudio de Optuna finalizado.
    - architecture_name (str): Nombre de la arquitectura utilizada.

    Returns:
    - pathlib.Path: Ruta del archivo JSON generado.
    """
    # Diccionario con la información del mejor trial
    output = {
        "architecture": architecture_name,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params
    }

    # Guardar el diccionario en formato JSON
    output_path = OPTUNA_DIR / f"{architecture_name}_best_params.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    return output_path


def main():
    """
    Ejecuta la búsqueda de hiperparámetros para la arquitectura seleccionada.

    La función procesa los argumentos de entrada, configura MLflow, ejecuta la optimización mediante Optuna,
    guarda el mejor trial en formato JSON y muestra un resumen de los mejores hiperparámetros encontrados.

    Returns:
    - None
    """
    # Leer los argumentos de la línea de comandos
    args = parse_args()

    # Configurar el experimento de MLflow
    setup_mlflow(experiment_name=args.experiment)

    # Obtener el modelo base y la función de preprocesamiento seleccionadas
    base_model_fn, preprocess_fn = MODELS[args.model]

    # Ejecutar la búsqueda de hiperparámetros
    study = run_hyperparameter_search(
        base_model_fn=base_model_fn,
        preprocess_fn=preprocess_fn,
        n_trials=args.trials,
    )

    # Guardar la información del mejor trial
    json_path = save_best_trial(
        study=study,
        architecture_name=base_model_fn.__name__
    )

    # Mostrar un resumen de los resultados obtenidos
    print(f"Mejor trial: {study.best_trial.number}")
    print(f"Mejor val_loss: {study.best_value:.6f}")

    print("\nMejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f" - {key}: {value}")

    print(f"\nParámetros guardados en: {json_path}")


if __name__ == "__main__":
    main()
