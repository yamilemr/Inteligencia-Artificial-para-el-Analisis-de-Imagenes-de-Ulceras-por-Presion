"""
Genera los archivos CSV con los metadatos y particiones (train, val y test) de las imágenes.

Este script procesa dos conjuntos de datos: uno propio (no disponible públicamente) clasificado
por un experto en úlceras por presión, y el Pressure Injury Images Dataset (PIID, disponible en
https://github.com/FU-MedicalAI/PIID), el cual fue reclasificado por el mismo experto.

El dataset principal (propio) se divide utilizando particionamiento estratificado por paciente para 
mantener el equilibrio de clases y evitar la fuga de datos. Los pacientes que sólo presentan 
piel sana se reservan para el conjunto de prueba, para complementar el dataset de PIID. 

Todas las imágenes de PIID se asignan al conjunto de prueba, ya que no es posible descartar 
la presencia de pacientes repetidos, lo que representaría un riesgo de fuga de datos.

Comando de ejecución (desde la raíz del proyecto):
    uv run python scripts/generate_metadata.py
"""

from upp_classification.metadata_generation import generate_piid_metadata, generate_upp_metadata, verify_data_leakage
from upp_classification.visualization import plot_dataset_distribution
from upp_classification.config import UPP_CSV_FILE, PIID_CSV_FILE, FIGURES_DIR


def main():
    """
    Genera los metadatos y particiones (train/val/test) para UPP y PIID
    y los guarda en archivos CSV.

    También imprime resúmenes de distribución de clases, ejecuta las
    pruebas para verificar que no exista fuga de datos y exporta una 
    gráfica con la distribución final.

    Returns:
    - None
    """
    print("\n--- Generar metadatos y particiones para UPP ---")
    df_upp = generate_upp_metadata()

    print(f"\nTotal de imágenes: {len(df_upp)}")
    print("Imágenes por clase:")
    print(df_upp["label"].value_counts().to_string())

    print(f"\nMetadatos guardados en: {UPP_CSV_FILE}")

    verify_data_leakage(df=df_upp)

    print("\n--- Generar metadatos para PIID ---")
    df_piid = generate_piid_metadata()

    print(f"\nTotal de imágenes: {len(df_piid)}")
    print("Imágenes por clase:")
    print(df_piid["label"].value_counts().to_string())

    print(f"\nMetadatos guardados en: {PIID_CSV_FILE}")

    fig = plot_dataset_distribution()
    output_path = FIGURES_DIR / "dataset_distribution.png"
    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"\nDistribución de imágenes por clase y split guardada en: {output_path}\n")

if __name__ == "__main__":
    main()
