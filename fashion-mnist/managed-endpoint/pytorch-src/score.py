"""Prediction."""

import json
import logging
import os

import numpy as np
import torch
from torch import Tensor, nn

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
    logging.info('Init started')

    global model
    global device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: %s', device)

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'weights.pth')

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logging.info('Init completed')


def run(raw_data):
    logging.info('Run started')

    x = json.loads(raw_data)['data']
    x = np.array(x).reshape((1, 1, 28, 28))
    x = torch.from_numpy(x).float().to(device)

    predicted_index = predict(model, x).item()
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
