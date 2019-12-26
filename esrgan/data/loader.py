import tensorflow as tf
from esrgan.utils.helpers import load_config

from tf.data.experimental import AUTOTUNE

def get_file_paths_ds(datadir: str, folder: str) -> tf.data.Dataset:
    subdir = ''
    while True:
        try:
            pattern = f'{datadir}/*/{folder}/*{subdir}/*.png'
            files_ds = tf.data.Dataset.list_files(pattern, shuffle=False)
            print('[-]', "Pattern matched:", pattern)
            return files_ds
        except tf.errors.InvalidArgumentError:
            print('[!]', "No matches found for pattern:", pattern)
            print('[-]', "Incrementing folder depth.")
            subdir += '/*'


def load_image_from_path(image_path: str) -> tf.Tensor:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    return image


def rotate(x: tf.Tensor, seed: int) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, 
                          tf.random_uniform(
                              shape=[], 
                              minval=0, 
                              maxval=4, 
                              dtype=tf.int32, 
                              seed=seed)
                          )


def flip(x: tf.Tensor, seed: int) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x, 
                                        seed=seed)
    return x


def get_augmented_ds_from_paths (paths_ds: tf.data.Dataset, seed: int) -> tf.data.Dataset:
    original_ds = paths_ds.map(load_image_from_path,
                               num_parallel_calls=AUTOTUNE)

    augmented_ds = tf.data.Dataset()
    augmented_ds.concatenate(original_ds)

    rotate_fn = lambda x: rotate(x, seed)
    flip_fn = lambda x: flip(x, seed)

    for map_func in [rotate_fn, flip_fn]:
        augmented_ds.concatenate(
            original_ds.map(
                map_func=map_func,
                num_parallel_calls=AUTOTUNE
            )
        )

    return augmented_ds

def load_datasets(config=load_config()) -> (tf.data.Dataset, tf.data.Dataset):
    def load_dataset(dataset):
        lrds = get_file_paths_ds(config['datadir'], f'{dataset}_lr')
        hrds = get_file_paths_ds(config['datadir'], f'{dataset}_hr')
        lrds = get_augmented_ds_from_paths(lrds, config['seed'])
        lrds = get_augmented_ds_from_paths(lrds, config['seed'])
        return tf.data.Dataset.zip(lrds, hrds)

    return load_dataset('train'), load_dataset('validation')
