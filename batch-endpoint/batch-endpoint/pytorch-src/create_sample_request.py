"""Creates a sample request to be used in prediction."""

from torchvision import datasets
import os


def main() -> None:
    """Creates a sample request to be used in prediction."""

    dir_name = 'batch-endpoint/sample-request'

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
    )

    os.makedirs(name=dir_name, exist_ok=True)
    for i, (image, _) in enumerate(test_data):
        if i == 200:
            break
        image.save(f'{dir_name}/{i+1:0>3}.png')


if __name__ == '__main__':
    main()
