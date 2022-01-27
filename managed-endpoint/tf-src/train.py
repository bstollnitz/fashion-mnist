"""Training and evaluation."""

import os
import time
from typing import Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

MODEL_DIRPATH = 'managed-endpoint/tf-model/weights'


def _get_data(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Downloads Fashion MNIST data, and returns two Dataset objects
    wrapping test and training data."""
    (training_images, training_labels), (
        test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (training_images, training_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels))

    train_dataset = train_dataset.map(lambda image, label:
                                      (float(image) / 255.0, label))
    test_dataset = test_dataset.map(lambda image, label:
                                    (float(image) / 255.0, label))

    train_dataset = train_dataset.batch(batch_size).shuffle(500)
    test_dataset = test_dataset.batch(batch_size).shuffle(500)

    return (train_dataset, test_dataset)


def _fit(dataset: tf.data.Dataset, model: tf.keras.Model,
         loss_fn: tf.keras.losses.Loss,
         optimizer: tf.optimizers.Optimizer) -> Tuple[float, float]:
    """Trains the given model for a single epoch."""
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    # Used for printing only.
    batch_count = len(dataset)
    print_every = 100

    for batch_index, (x, y) in enumerate(dataset):
        x = tf.cast(x, tf.float64)
        y = tf.cast(y, tf.int64)

        (y_prime, loss) = _fit_one_batch(x, y, model, loss_fn, optimizer)

        correct_item_count += (tf.math.argmax(y_prime,
                                              axis=1) == y).numpy().sum()
        loss_sum += loss.numpy()
        item_count += len(x)

        # Printing progress.
        if ((batch_index + 1) % print_every == 0) or ((batch_index + 1)
                                                      == batch_count):
            accuracy = correct_item_count / item_count
            average_loss = loss_sum / item_count
            print(f'[Batch {batch_index + 1:>3d} - {item_count:>5d} items] ' +
                  f'loss: {average_loss:>7f}, ' +
                  f'accuracy: {accuracy*100:>0.1f}%')

    average_loss = loss_sum / item_count
    accuracy = correct_item_count / item_count

    return (average_loss, accuracy)


@tf.function
def _fit_one_batch(
        x: tf.Tensor, y: tf.Tensor, model: tf.keras.Model,
        loss_fn: tf.keras.losses.Loss, optimizer: tf.keras.optimizers.Optimizer
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Trains a single minibatch (backpropagation algorithm)."""
    with tf.GradientTape() as tape:
        y_prime = model(x, training=True)
        loss = loss_fn(y, y_prime)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return (y_prime, loss)


def _evaluate(dataset: tf.data.Dataset, model: tf.keras.Model,
              loss_fn: tf.keras.losses.Loss) -> Tuple[float, float]:
    """Evaluates the given model for the whole dataset once."""
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    for (x, y) in dataset:
        x = tf.cast(x, tf.float64)
        y = tf.cast(y, tf.int64)

        (y_prime, loss) = _evaluate_one_batch(x, y, model, loss_fn)

        correct_item_count += (tf.math.argmax(
            y_prime, axis=1).numpy() == y.numpy()).sum()
        loss_sum += loss.numpy()
        item_count += len(x)

    average_loss = loss_sum / item_count
    accuracy = correct_item_count / item_count
    return (average_loss, accuracy)


@tf.function
def _evaluate_one_batch(
        x: tf.Tensor, y: tf.Tensor, model: tf.keras.Model,
        loss_fn: tf.keras.losses.Loss) -> Tuple[tf.Tensor, tf.Tensor]:
    """Evaluates a single minibatch."""
    y_prime = model(x, training=False)
    loss = loss_fn(y, y_prime)

    return (y_prime, loss)


def training_phase():
    """Trains the model for a number of epochs, and saves it."""
    learning_rate = 0.1
    batch_size = 64
    epochs = 5

    (train_dataset, test_dataset) = _get_data(batch_size)
    # _visualize_data(train_dataset)

    model = NeuralNetwork()
    # model.build((1, 28, 28))
    # model.summary()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.optimizers.SGD(learning_rate)

    print('\n***Training***')
    t_begin = time.time()

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}\n-------------------------------')
        (train_loss, train_accuracy) = _fit(train_dataset, model, loss_fn,
                                            optimizer)
        print(f'Train loss: {train_loss:>8f}, ' +
              f'train accuracy: {train_accuracy * 100:>0.1f}%')

    t_elapsed = time.time() - t_begin
    print(f'\nTime per epoch: {t_elapsed / epochs :>.3f} sec')

    print('\n***Evaluating***')
    (test_loss, test_accuracy) = _evaluate(test_dataset, model, loss_fn)
    print(f'Test loss: {test_loss:>8f}, ' +
          f'test accuracy: {test_accuracy * 100:>0.1f}%')

    model.save_weights(MODEL_DIRPATH)


def main() -> None:
    training_phase()


if __name__ == '__main__':
    main()
