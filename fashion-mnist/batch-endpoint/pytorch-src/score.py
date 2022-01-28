"""Prediction."""

import argparse
import logging
import os

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms

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


def predict(model: nn.Module, x: Tensor) -> torch.Tensor:
    with torch.no_grad():
        y_prime = model(x)
        probabilities = nn.functional.softmax(y_prime, dim=1)
        predicted_indices = probabilities.argmax(1)
    return predicted_indices


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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info('Device: %s', device)

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'weights.pth')
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logger.info('Init completed')


def run(mini_batch):
    logger.info('run(%s started: %s', mini_batch, {__file__})
    predicted_names = []
    transform = transforms.ToTensor()

    for image_path in mini_batch:
        image = Image.open(image_path)
        tensor = transform(image).to(device)
        predicted_index = predict(model, tensor).item()
        predicted_names.append(f'{image_path}: {labels_map[predicted_index]}')

    logger.info('Run completed')
    return predicted_names
