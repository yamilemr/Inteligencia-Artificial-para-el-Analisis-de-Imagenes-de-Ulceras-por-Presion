import numpy as np
import tensorflow as tf
from pathlib import Path
from upp_classification.data_loader import prepare_image
from upp_classification.config import IMAGE_SIZE, CLASS_NAMES


def predict_single_image(model, image_path, image_size=IMAGE_SIZE):
    """
    Realiza la predicción para una sola imagen utilizando un modelo entrenado.

    Args:
    - model (keras.Model): Modelo entrenado (con el preprocesamiento integrado).
    - image_path (str o Path): Ruta de la imagen a predecir.
    - image_size (tuple, optional): Tamaño para redimensionar la imagen (alto, ancho). 
                                    Por defecto es IMAGE_SIZE.

    Returns:
    - dict: Diccionario con los resultados de la inferencia:
            - predicted_class_id (int): Identificador numérico de la clase predicha.
            - predicted_class_name (str): Nombre descriptivo de la clase predicha.
            - probabilities (dict): Diccionario con las probabilidades para cada clase.
    
    Raises:
    - FileNotFoundError: Si el archivo de imagen no existe en la ruta proporcionada.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"No se encontró la imagen en la ruta: {image_path}")

    # Leer la imagen, decodificarla, redimensionarla y convertir a float32
    image = prepare_image(image_path=str(image_path), label=None, image_size=image_size)
    
    # Agregar la dimensión del batch (1, H, W, C)
    image = tf.expand_dims(image, axis=0)

    # Obtener los logits predichos por el modelo
    logits = model(image, training=False)

    # Convertir los logits en probabilidades usando softmax
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    # Obtener el índice de la clase predicha (la de mayor probabilidad)
    predicted_class_id = int(np.argmax(probs))

    # Obtener el nombre descriptivo de la clase predicha
    predicted_class_name = CLASS_NAMES[predicted_class_id]

    # Diccionario con las probabilidades de cada clase
    class_probabilities = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    
    # Construir el diccionario de resultados
    result = {
        "predicted_class_id": predicted_class_id,
        "predicted_class_name": predicted_class_name,
        "probabilities": class_probabilities
    }
    
    return result
