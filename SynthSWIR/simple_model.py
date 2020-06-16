"""Architecture of the small ML model."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    """Create the ML model."""
    model = keras.Sequential([
        layers.Dense(12, activation='relu', input_shape=(4,)),
        layers.Dense(12, activation='selu'),
        layers.Dense(12, activation='relu'),
        layers.Dense(2)
      ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model
