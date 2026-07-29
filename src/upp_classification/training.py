from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from upp_classification.mlflow_tracking import MLflowMetricsCallback
from upp_classification.config import EPOCHS


def train_model(model, train_ds, val_ds, class_weights, epochs=EPOCHS, use_mlflow=True, pruning_callback=None):
    """
    Ejecuta el ciclo de entrenamiento de un modelo de Keras.

    Args:
    - model (tensorflow.keras.Model): Modelo de Keras previamente construido y compilado.
    - train_ds (tf.data.Dataset): Conjunto de datos de entrenamiento agrupado en lotes.
    - val_ds (tf.data.Dataset): Conjunto de datos de validación agrupado en lotes.
    - class_weights (dict): Diccionario con los pesos balanceados de cada clase, calculados a partir 
                            del conjunto de entrenamiento.
    - epochs (int, optional): Número máximo de épocas para el entrenamiento. Por defecto es EPOCHS.
    - use_mlflow (bool, optional): Indica si se debe incluir el callback para registrar las métricas 
                                   generadas por época en MLflow. Por defecto es True.
    - pruning_callback (keras.callbacks.Callback, optional): Callback de Optuna utilizado para detener 
                                                             anticipadamente trials poco prometedores.
                                                             Por defecto es None.

    Returns:
    - keras.callbacks.History: Objeto de Keras que contiene el registro de los valores de pérdida 
                               y métricas calculados a lo largo de las épocas.
    """
    callbacks = []
    
    # Callback que registra en MLflow las métricas generadas al finalizar cada época de entrenamiento
    if use_mlflow:
        callbacks.append(MLflowMetricsCallback())
    
    # Callback de Optuna para podar trials con bajo rendimiento (sólo durante la optimización)
    if pruning_callback:
        callbacks.append(pruning_callback)
        
    # Callback para detener el entrenamiento cuando el modelo deje de mejorar
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=10, # Espera 10 épocas sin mejoras antes de detenerse
            min_delta=1e-4, # Considera una mejora sólo si val_loss disminuye al menos 1e-4
            restore_best_weights=True # Restaura los pesos de la época con el mejor val_loss
        )
    )

    # Callback para reducir la tasa de aprendizaje cuando el modelo se estanque
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.6, # Factor por el que se multiplica la tasa de aprendizaje al reducirse
            patience=3, # Número de épocas que espera sin mejoras antes de reducir
            min_delta=1e-4, # Considera una mejora sólo si val_loss disminuye al menos 1e-4
            min_lr=1e-7 # Learning rate mínimo permitido
        )
    )

    # Entrenar el modelo
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        shuffle=False, # El shuffle ya se realiza al cargar los datos con tf.data.Dataset
        class_weight=class_weights, # Da mayor peso a las clases minoritarias durante el cálculo de la pérdida
        callbacks=callbacks
    )

    return history
