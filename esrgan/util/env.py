"""
Module for setting up and describing Tensorflow environment.
"""

import tensorflow as tf
from tensorflow.python.client import device_lib

import yaml

def setup_environment():
    '''Setup Tensorflow Environment'''
    print('Tensorflow Version:', tf.__version__)
    tf.compat.v1.disable_eager_execution()


def show_available_devices():
    '''Display List of Available Devices'''
    local_device_protos = device_lib.list_local_devices()
    print(local_device_protos)
