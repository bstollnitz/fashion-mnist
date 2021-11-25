"""Creates a sample request to be used in prediction."""

import json
import os

from train import _get_data


def create_sample_request() -> None:
    """Creates a sample request to be used in prediction."""
    batch_size = 64
    dir_name = 'managed-endpoint/sample-request'
    file_name = dir_name + '/sample_request.json'
    (_, test_dataloader) = _get_data(batch_size)

    (x_batch, _) = next(iter(test_dataloader))
    x = x_batch[0, 0, :, :].cpu().numpy().tolist()

    os.makedirs(name=dir_name, exist_ok=True)
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump({'data': x}, file)


def main() -> None:
    create_sample_request()


if __name__ == '__main__':
    main()
