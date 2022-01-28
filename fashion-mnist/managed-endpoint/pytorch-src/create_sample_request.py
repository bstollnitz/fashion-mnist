"""Creates a sample request to be used in prediction."""

import json
import os
from pathlib import Path

from train import _get_data

DATA_PATH = 'fashion-mnist/managed-endpoint/data'
SAMPLE_REQUEST = 'fashion-mnist/managed-endpoint/sample-request'


def create_sample_request() -> None:
    """Creates a sample request to be used in prediction."""
    batch_size = 64
    (_, test_dataloader) = _get_data(batch_size)

    (x_batch, _) = next(iter(test_dataloader))
    x = x_batch[0, 0, :, :].cpu().numpy().tolist()

    os.makedirs(name=SAMPLE_REQUEST, exist_ok=True)
    with open(Path(SAMPLE_REQUEST, 'sample_request.json'),
              'w',
              encoding='utf-8') as file:
        json.dump({'data': x}, file)


def main() -> None:
    create_sample_request()


if __name__ == '__main__':
    main()
