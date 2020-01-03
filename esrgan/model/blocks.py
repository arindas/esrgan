"""
Module for defining all the blocks to be used in building the network.
"""

import tensorflow as tf

class ResidualDenseBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, use_bias=False):
        super(ResidualDenseBlock, self).__init__()

    def call(self, input_tensor, training=False):
        pass

class RRDB(tf.keras.Model):
    def __init__(self):
        super(RRDB, self).__init__()

    def call(self, input_tensor, training=False):
        pass
