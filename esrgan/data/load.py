"""
Module for loading datasets sepcific to esrgan.
"""

import tensorflow as tf
import yaml

import esrgan

from esrgan.util.dataset import (
    get_file_paths_ds,
    get_augmented_ds_from_paths
)
from esrgan.data.augment import (
    horizontal_flip,
    rotate
)


def load_config(config_file="config.yaml"):
    """Loads a config dictionary from a yaml file.

    Keyword arguments:
    config_file -- path to yaml config file
    """
    with open(config_file, 'r') as stream:
        return yaml.safe_load(stream)


def load_image_from_path(image_path: str) -> tf.Tensor:
    """Loads an image from path

    Keyword arguments:
    image_path -- image path
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    return image


def load_dataset_segment(segment: str,
                         config=load_config()) -> tf.data.Dataset:
    """Loads a dataset segment (train or validation) from esrgan
    dataset dir, while adhering to the given config.

    Keyword arguments:
    segment -- segment to load (train or validation)
    config -- loaded config dict
    """
    lrds = get_file_paths_ds(config['datadir'], f'{segment}_lr', 'png')
    hrds = get_file_paths_ds(config['datadir'], f'{segment}_hr', 'png')
    augmentations = [rotate, horizontal_flip]

    lrds = get_augmented_ds_from_paths(lrds,
                                       load_image_from_path,
                                       augmentations)
    hrds = get_augmented_ds_from_paths(hrds,
                                       load_image_from_path,
                                       augmentations)
    return tf.data.Dataset.zip(lrds, hrds)


def load_datasets(fetch=True,
                  config=load_config()) -> (tf.data.Dataset, tf.data.Dataset):
    """Load the training and validation dataset for esrgan.

    Keyword arguments:
    config -- loaded config dict
    """

    if fetch:
        esrgan.data.fetch.fetch_datasets(config)

    esrgan.data.augment.SEED = config['seed']
    train_ds = load_dataset_segment('train', config)
    validation_ds = load_dataset_segment('validation', config)
    return train_ds, validation_ds
