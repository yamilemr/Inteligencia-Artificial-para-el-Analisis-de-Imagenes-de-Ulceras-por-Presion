import os
import re
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from upp_classification.config import UPP_IMGS_DIR, UPP_CSV_FILE, PIID_IMGS_DIR, PIID_CSV_FILE, SEED


def generate_piid_metadata(imgs_dir=PIID_IMGS_DIR, csv_output_path=PIID_CSV_FILE):
    """
    Genera los metadatos para el dataset público PIID. Todas las imágenes de PIID se asignan
    directamente al conjunto de prueba, debido a que no se sabe si presentan fuga de datos.

    Las imágenes deben estar almacenadas en una sola carpeta y deben seguir el formato de 
    nombre indicado en la expresión regular:
    piid_categoria(i, ii, iii, iv o nc)_numimg.(jpg o jpeg)

    Args:
    - imgs_dir (str o Path, optional): Ruta del directorio que contiene las imágenes de PIID. 
                                       Por defecto es PIID_IMGS_DIR.
    - csv_output_path (str o Path, optional): Ruta donde se guardará el archivo CSV con los metadatos 
                                              de las imágenes. Por defecto es UPP_CSV_FILE.

    Returns:
    - pd.DataFrame: DataFrame con los metadatos generados para las imágenes de PIID.
    """
    patron_piid = re.compile(
        r'^piid_(i|ii|iii|iv|nc)_(img\d{3})\.(jpg|jpeg)$',
        re.IGNORECASE
    )

    rows_piid = []

    # Recorrer todas las imágenes de la carpeta
    for filename in sorted(os.listdir(imgs_dir)):
        # Verificar que el nombre del archivo cumpla con el patrón
        match = patron_piid.match(filename)

        if not match:
            print(f"Nombre inválido: {filename}")
            continue

        # Agregar la información de la imagen a rows_piid
        rows_piid.append({
            "filename": filename,
            "label": match.group(1).lower(),
            "split": "test"
        })

    # Crear el df con la información de las imágenes y guardarlo en un CSV
    df_piid = pd.DataFrame(rows_piid)
    df_piid.to_csv(csv_output_path, index=False)
    
    return df_piid


def generate_upp_metadata(imgs_dir=UPP_IMGS_DIR, csv_output_path=UPP_CSV_FILE):
    """
    Lee las imágenes de UPP, extrae sus metadatos y realiza la partición train/val/test.
    
    Para test se asignan los pacientes que únicamente presentan piel sana, ya que esta clase 
    no está presente en el dataset público PIID.

    Para la partición de train/val se utiliza StratifiedGroupKFold, para evitar fuga de datos 
    (garantiza que un paciente no aparezca en ambos conjuntos) y buscar una distribución similar
    de las clases.

    Las imágenes deben estar almacenadas en una sola carpeta y deben seguir el formato de
    nombre indicado en la expresión regular:
    idpaciente_idlesion_categoria(i, ii, iii, iv, nc o ps)_numimg.(jpg o jpeg)

    Args:
    - imgs_dir (str o Path, optional): Ruta del directorio que contiene las imágenes de UPP. 
                                       Por defecto es UPP_IMGS_DIR.
    - csv_output_path (str o Path, optional): Ruta donde se guardará el archivo CSV con los metadatos 
                                              de las imágenes. Por defecto es UPP_CSV_FILE.

    Returns:
    - pd.DataFrame: DataFrame con los metadatos generados y las particiones asignadas.
    """
    patron_upp = re.compile(
        r'^(p\d{3})_(u\d{2}|ps\d{2})_(ps|i|ii|iii|iv|nc|ps)_(img\d{2})\.(jpg|jpeg)$',
        re.IGNORECASE
    )
    
    rows_upp = []

    # Recorrer todas las imágenes de la carpeta
    for filename in sorted(os.listdir(imgs_dir)):
        # Verificar que el nombre del archivo cumpla con el patrón
        match = patron_upp.match(filename)

        if not match:
            print(f"Nombre inválido: {filename}")
            continue

        # Extraer el identificador del paciente y de la lesión
        patient_id = match.group(1).lower()
        lesion_id = match.group(2).lower()

        # Agregar la información de la imagen a rows_upp
        rows_upp.append({
            "filename": filename,
            "label": match.group(3).lower(),
            "patient_id": patient_id,
            "lesion_id": f"{patient_id}_{lesion_id}",
            "split": None
        })

    # Crear el df con la información de las imágenes
    df_upp = pd.DataFrame(rows_upp)

    # Etiquetas presentes por cada paciente
    clases_por_paciente = df_upp.groupby("patient_id")["label"].unique()

    # Asignar a test los pacientes que sólo tienen piel sana (ps)
    pacientes_ps = clases_por_paciente[
        clases_por_paciente.apply(lambda x: set(x).issubset({"ps"}))
    ].index
    df_upp.loc[df_upp["patient_id"].isin(pacientes_ps), "split"] = "test"

    # Filtrar las imágenes que aún no pertenecen a ningún split
    new_df = df_upp[df_upp["split"].isna()]

    X = new_df["filename"]
    y = new_df["label"]
    groups = new_df["patient_id"]

    # Objeto para realizar StratifiedGroupKFold
    # Los datos se dividen en 5 folds (grupos) de pacientes, con una distribución similar de las clases
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Obtiene la primera partición generada por por StratifiedGroupKFold
    # train_idx contiene los índices de las filas de train (4 folds)
    # val_idx contie los índices de las filas de validation (1 fold)
    train_idx, val_idx = next(sgkf.split(X=X, y=y, groups=groups))

    # Asignar los splits de train y validation al df original
    df_upp.loc[new_df.iloc[train_idx].index, "split"] = "train"
    df_upp.loc[new_df.iloc[val_idx].index, "split"] = "val"

    # Guardar el df en un CSV
    df_upp.to_csv(csv_output_path, index=False)
    
    return df_upp


def verify_data_leakage(df):
    """
    Imprime un resumen de los conjuntos train, val y test para verificar que ningún
    paciente está presente en más de una partición de forma simultánea.

    Args:
    - df (pd.DataFrame): DataFrame que contiene los metadatos de las imágenes.
                         Debe tener las columnas 'patient_id' y 'split'.

    Returns:
    - None: La función imprime los resultados en consola.
    """
    pacientes_train = set(df[df["split"] == "train"]["patient_id"].unique())
    pacientes_val = set(df[df["split"] == "val"]["patient_id"].unique())
    pacientes_test = set(df[df["split"] == "test"]["patient_id"].unique())

    print("\n--- Verificación de Data Leakage ---")

    # Pacientes por conjunto
    print(f"\nPacientes en train: {len(pacientes_train)}")
    print(pacientes_train)

    print(f"\nPacientes en val: {len(pacientes_val)}")
    print(pacientes_val)

    print(f"\nPacientes en test: {len(pacientes_test)}")
    print(pacientes_test)

    # Verificar que no hay pacientes compartidos
    print(f"\nTrain ∩ Val: {len(pacientes_train & pacientes_val)}")
    print(f"Train ∩ Test: {len(pacientes_train & pacientes_test)}")
    print(f"Val ∩ Test: {len(pacientes_val & pacientes_test)}")
