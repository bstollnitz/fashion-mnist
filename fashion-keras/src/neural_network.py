"""Neural network class."""

import tensorflow as tf


class NeuralNetwork(tf.keras.Model):
    """Neural network that classifies Fashion MNIST-style images."""

    def __init__(self):
        super().__init__()
        self.sequence = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y_prime = self.sequence(x)
        return y_prime
