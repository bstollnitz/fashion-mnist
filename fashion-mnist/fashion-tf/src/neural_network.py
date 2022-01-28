"""Neural network class."""

import tensorflow as tf


class NeuralNetwork(tf.keras.Model):
    """Neural network that classifies Fashion MNIST-style images."""

    def __init__(self):
        super().__init__()
        initializer = tf.keras.initializers.GlorotUniform()
        self.w1 = tf.Variable(initializer(shape=(784, 20)))
        self.b1 = tf.Variable(tf.zeros(shape=(20,)))
        self.w2 = tf.Variable(initializer(shape=(20, 10)))
        self.b2 = tf.Variable(tf.zeros(shape=(10,)))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.reshape(x, [-1, 784])
        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.w2) + self.b2
        return x
