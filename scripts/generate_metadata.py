"""
Genera los archivos CSV con los metadatos y particiones (train, val y test) de las imágenes.

Este script procesa dos conjuntos de datos: uno propio (no disponible públicamente) clasificado
por un experto en úlceras por presión, y el Pressure Injury Images Dataset (PIID, disponible en
https://github.com/FU-MedicalAI/PIID), el cual fue reclasificado por el mismo experto.

El dataset principal (propio) se divide utilizando particionamiento estratificado por paciente  
para mantener el equilibrio de clases y evitar la fuga de datos. Los sujetos que sólo presentan 
piel sana se reservan para el conjunto de prueba, para complementar el dataset de PIID. 

Todas las imágenes de PIID se asignan al conjunto de prueba, ya que no es posible descartar 
la presencia de pacientes repetidos, lo que representaría un riesgo de fuga de datos.

Comando de ejecución (desde la raíz del proyecto):
    uv run python scripts/generate_metadata.py
"""

from upp_classification.metadata_generation import generate_piid_metadata, generate_upp_metadata
from upp_classification.visualization import plot_class_counts_per_dataset, plot_class_split_distribution, plot_patient_split_assignment
from upp_classification.config import UPP_CSV_FILE, PIID_CSV_FILE, DATASET_PLOTS_DIR


def main():
    """
    Ejecuta el flujo de generación de metadatos y particiones (train/val/test) 
    para los datasets UPP y PIID, guardando los resultados en archivos CSV.

    Adicionalmente, genera y exporta gráficas sobre la distribución de clases 
    y una matriz de asignación de pacientes para verificar visualmente 
    la ausencia de fuga de datos.

    Returns:
    - None
    """
    # Generar los metadatos
    generate_upp_metadata()
    generate_piid_metadata()

    # Gráficos de barras con el número de imágenes por clase
    fig_upp, fig_piid = plot_class_counts_per_dataset()

    path_upp = DATASET_PLOTS_DIR / "upp_distribution.png"
    fig_upp.savefig(path_upp, dpi=500, bbox_inches="tight")

    path_piid = DATASET_PLOTS_DIR / "pidd_distribution.png"
    fig_piid.savefig(path_piid, dpi=500, bbox_inches="tight")

    # Gráfico de barras con la distribución de imágenes por clase y partición
    fig_splits = plot_class_split_distribution()
    path_splits = DATASET_PLOTS_DIR / "class_split_distribution.png"
    fig_splits.savefig(path_splits, dpi=500, bbox_inches="tight")

    # Gráfico con la asignación de cada paciente a una única partición
    fig_patients = plot_patient_split_assignment()
    path_patients = DATASET_PLOTS_DIR / "patient_split_assignment.png"
    fig_patients.savefig(path_patients, dpi=500, bbox_inches="tight")

    # Imprimir resumen
    print(f"\nMetadatos de UPP guardados en: {UPP_CSV_FILE}")
    print(f"Distribución de clases de UPP guardada en: {path_upp}")

    print(f"\nMetadatos de PIID guardados en: {PIID_CSV_FILE}")
    print(f"Distribución de clases de PIID guardada en: {path_piid}")

    print(f"\nDistribución por clase y partición guardada en: {path_splits}")
    print(f"Asignación de pacientes por partición guardada en: {path_patients}\n")


if __name__ == "__main__":
    main()
