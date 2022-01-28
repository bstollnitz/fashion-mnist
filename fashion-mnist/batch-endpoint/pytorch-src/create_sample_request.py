"""Creates a sample request to be used in prediction."""

from torchvision import datasets
import os

DATA_PATH = 'fashion-mnist/batch-endpoint/data'
SAMPLE_REQUEST = 'fashion-mnist/batch-endpoint/sample-request'


def main() -> None:
    """Creates a sample request to be used in prediction."""

    test_data = datasets.FashionMNIST(
        root=DATA_PATH,
        train=False,
        download=True,
    )

    os.makedirs(name=SAMPLE_REQUEST, exist_ok=True)
    for i, (image, _) in enumerate(test_data):
        if i == 200:
            break
        image.save(f'{SAMPLE_REQUEST}/{i+1:0>3}.png')


if __name__ == '__main__':
    main()
