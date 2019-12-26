import tensorflow as tf


def get_file_paths_ds(datadir, folder):
    subdir = ''
    while True:
        try:
            pattern = f'{datadir}/*/{folder}/*{subdir}/*.png'
            files_ds = tf.data.Dataset.list_files(pattern, shuffle=False)
            print ('[-]', "Pattern matched:", pattern)
            return files_ds
        except tf.errors.InvalidArgumentError as e:
            print ('[!]', "No matches found for pattern:", pattern)
            print ('[-]', "Incrementing folder depth.")
            subdir += '/*'


def map_paths_to_images (lr_image_path, hr_image_path):
    lr_image, hr_image = tf.io.read_file(lr_image_path),\
        tf.io.read_file(hr_image_path)
    lr_image, hr_image = tf.image.decode_png(lr_image, channels=3),\
        tf.image.decode_png(hr_image, channels=3)
    return lr_image, hr_image
