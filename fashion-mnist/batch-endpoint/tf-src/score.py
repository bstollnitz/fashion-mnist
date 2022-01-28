"""Prediction."""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image

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


def init():
    global logger
    global model
    global device

    arg_parser = argparse.ArgumentParser(description='Argument parser.')
    arg_parser.add_argument('--logging_level', type=str, help='logging level')
    args, _ = arg_parser.parse_known_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(args.logging_level.upper())

    logger.info('Init started')

    physical_devices = tf.config.list_physical_devices('GPU')
    logger.info('Num GPUs: %d', len(physical_devices))

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'tf-model')
    model = tf.keras.models.load_model(model_path, compile=False)

    logger.info('Init completed')


def run(mini_batch):
    logger.info('Run started')
    predicted_names = []

    for image_path in mini_batch:
        image = Image.open(image_path)
        array = tf.keras.preprocessing.image.img_to_array(image).reshape(
            (-1, 28, 28))
        predicted_index = np.argmax(model.predict(array))
        predicted_names.append(f'{image_path}: {labels_map[predicted_index]}')

    logger.info('Run completed')
    return predicted_names
