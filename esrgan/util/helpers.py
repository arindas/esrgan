import tensorflow as tf
from tensorflow.python.client import device_lib

import yaml

def setup_environment():
    '''Setup Tensorflow Environment'''
    print('Tensorflow Version:', tf.__version__)
    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos if x.device_type == 'GPU'])
    tf.compat.v1.disable_eager_execution()


def show_available_devices():
    '''Display List of Available Devices'''
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as stream:
        return yaml.safe_load(stream)

