"""
Extrae todos los parámetros, métricas y tags registrados en todos los experimentos 
almacenados en la base de datos de MLflow y los exporta a un archivo CSV. Asimismo, 
se extraen los historiales de las métricas de entrenamiento en formato Parquet.

Comando de ejecución (desde la raíz del proyecto):
    uv run python scripts/export_mlflow_data.py
"""

import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from upp_classification.config import MLFLOW_TRACKING_URI, METRICS_DIR, BEST_MODELS_DIR


def extract_all_mlflow_runs(tracking_uri=MLFLOW_TRACKING_URI):
    """
    Recupera todas las ejecuciones (runs) de todos los experimentos en MLflow,
    incluyendo sus parámetros, métricas y tags.

    Args:
    - tracking_uri (str, optional): URI de conexión al backend de tracking de MLflow.
                                    Por defecto es MLFLOW_TRACKING_URI.

    Returns:
    - pd.DataFrame: DataFrame con todos los datos extraídos de MLflow.
    """
    # Configurar el URI de tracking
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Obtener todos los experimentos existentes
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

    # Mapeo de experiment_id para buscar los runs asociados a cada experimento
    exp_ids = [exp.experiment_id for exp in experiments]

    # Extraer todos los runs de todos los experimentos
    # mlflow.search_runs procesa automáticamente params, metrics y tags (sin artefactos)
    df_runs = mlflow.search_runs(
        experiment_ids=exp_ids,
        run_view_type=mlflow.entities.ViewType.ALL
    )

    # Eliminar columnas
    cols_to_drop = [
        "artifact_uri",
        "tags.mlflow.loggedArtifacts",
        "tags.mlflow.source.git.repoURL",
        "tags.mlflow.source.git.commit",
        "tags.mlflow.source.name",
        "tags.mlflow.source.git.branch",
        "tags.mlflow.source.type",
        "tags.mlflow.user"
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df_runs.columns]
    
    df_runs = df_runs.drop(columns=cols_to_drop)

    return df_runs


def extract_metric_history_per_epoch(tracking_uri=MLFLOW_TRACKING_URI):
    """
    Extrae el historial época por época de todas las métricas registradas con step.

    Args:
    - tracking_uri (str, optional): URI de conexión al backend de tracking de MLflow.
                                    Por defecto es MLFLOW_TRACKING_URI.

    Returns:
    - pd.DataFrame: DataFrame con las métricas por época de todas las ejecuciones.
    """
    # Configurar el URI de tracking
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Obtener todos los experimentos existentes
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

    # Lista para almacenar los registros del historial
    history_data = []

    # Definir las métricas que se van a extraer del historial
    target_metrics = {"accuracy", "val_accuracy", "loss", "val_loss"}

    # Iterar sobre todos los runs de cada experimento
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            run_view_type=mlflow.entities.ViewType.ALL
        )

        for run in runs:
            run_id = run.info.run_id
            run_name = run.info.run_name

            # Recuperar los tags "stage" y "architecture"
            stage = run.data.tags.get("stage")
            architecture = run.data.tags.get("architecture")

            # Iterar sobre las métricas registradas en el run
            for metric_key in run.data.metrics.keys():
                if metric_key in target_metrics:
                    # Obtener el historial completo de la métrica
                    history = client.get_metric_history(run_id=run_id, key=metric_key)

                    # Almacenar cada valor del historial con su época correspondiente
                    for m in history:
                        history_data.append({
                            "experiment_id": exp.experiment_id,
                            "run_id": run_id,
                            "run_name": run_name,
                            "architecture": architecture,
                            "stage": stage,
                            "step": m.step,
                            "metric_name": metric_key,
                            "value": m.value
                        })

    return pd.DataFrame(history_data)


def main():
    """
    Ejecuta la extracción y exportación de los datos de MLflow.

    Returns:
    - None
    """
    # Extraer resumen completo de runs (parámetros, métricas y tags)
    df_runs = extract_all_mlflow_runs()

    if not df_runs.empty:
        output_file = METRICS_DIR / "mlflow_runs_export.csv"
        df_runs.to_csv(output_file, index=False)
        print(f"\nDatos de los runs de MLflow guardados en: {output_file}")

    # Extraer historial de métricas por época
    df_history = extract_metric_history_per_epoch()
    
    if not df_history.empty:
        history_output_file = METRICS_DIR / "mlflow_epoch_metrics.parquet"
        df_history.to_parquet(history_output_file, index=False)
        print(f"\nHistorial de métricas por época guardado en: {history_output_file}")

        # Filtrar y guardar el historial de los mejores modelos
        df_best_models = df_history[df_history["stage"] == "best_model_training"]

        if not df_best_models.empty:
            best_models_file = BEST_MODELS_DIR / "best_models_history.parquet"
            df_best_models.to_parquet(best_models_file, index=False)
            print(f"Historial sólo de los mejores modelos guardado en: {best_models_file}\n")


if __name__ == "__main__":
    main()
    