"""
Module for providing image augmentation operations.
"""

import tensorflow as tf

SEED = 1


def rotate(img: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(img,
                          tf.random.uniform(
                              shape=[],
                              minval=0,
                              maxval=4,
                              dtype=tf.int32,
                              seed=SEED)
                          )


def horizontal_flip(img: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    return tf.image.random_flip_left_right(img,
                                           seed=SEED)
