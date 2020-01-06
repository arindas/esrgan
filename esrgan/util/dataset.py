"""
Module containing utility functions for loading and augmenting datasets.
"""

import tensorflow as tf


def get_file_paths_ds(datadir: str, folder: str, ext: str) -> tf.data.Dataset:
    """Creates a file paths dataset from a given root datadir,
    a specific subfolder and an extension.

    Keyword arguments:
    datadir -- root datadir of all datasets
    folder -- subfolder to walk in each dataset directory
    ext -- file extensions to consider
    """
    subdir = ''
    while True:
        try:
            pattern = f'{datadir}/*/{folder}/*{subdir}/*.{ext}'
            files_ds = tf.data.Dataset.list_files(pattern, shuffle=False)
            print('[-]', "Pattern matched:", pattern)
            return files_ds
        except tf.errors.InvalidArgumentError:
            print('[!]', "No matches found for pattern:", pattern)
            print('[-]', "Incrementing folder depth.")
            subdir += '/*'


def get_augmented_ds_from_paths(paths_ds: tf.data.Dataset, load_func,
                                augmentations: []) -> tf.data.Dataset:
    """Creates an augmented dataset from a file paths dataset.

    Keyword arguments:
    paths_ds -- file paths dataset
    load_func -- callable to load image from path
    augmentations -- list of callable augmentation functions
    """
    original_ds = paths_ds.map(load_func, tf.data.experimental.AUTOTUNE)
    augmented_ds = original_ds.take(-1)

    for map_func in augmentations:
        augmented_ds.concatenate(
            original_ds.map(
                map_func=map_func,
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        )

    return augmented_ds
