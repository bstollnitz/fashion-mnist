"""Prediction."""

import json
import logging
import os

import numpy as np
import tensorflow as tf

from neural_network import NeuralNetwork

labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}


@tf.function
def predict(model: tf.keras.Model, x: np.ndarray) -> tf.Tensor:
    y_prime = model(x, training=False)
    probabilities = tf.nn.softmax(y_prime, axis=1)
    predicted_indices = tf.math.argmax(input=probabilities, axis=1)
    return predicted_indices


def init():
    logging.info('Init started')

    global model
    global device

    physical_devices = tf.config.list_physical_devices('GPU')
    logging.info('Num GPUs: %d', len(physical_devices))

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'tf-model')
    model = tf.keras.models.load_model(model_path, compile=False)

    logging.info('Init completed')


def run(raw_data):
    logging.info('Run started')

    x = json.loads(raw_data)['data']
    x = np.array(x).reshape((-1, 28, 28))

    predicted_index = np.argmax(model.predict(x))
    predicted_name = labels_map[predicted_index]

    logging.info('Predicted name: %s', predicted_name)

    logging.info('Run completed')
    return predicted_name


if __name__ == '__main__':
    init()
    with open('managed-endpoint/sample-request/sample_request.json',
              encoding='utf-8') as file:
        raw_data = file.read()
    print(run(raw_data))
