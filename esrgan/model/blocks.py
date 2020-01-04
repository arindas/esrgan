"""
Module for defining all the blocks to be used in building the network.
"""

import tensorflow as tf


class ConvLReLU(tf.keras.Model):
    """Conv LeakyReLU pair.

    Elementary block inside ResidualDenseBlock. It comprises of
    a 2D Convolution followed by a leaky ReLu activation.

    Arguments:
    filters: number of output channels
    use_bias: refer tf.keras.layers.Conv2D
    """
    def __init__(self, filters=32, use_bias=True):
        super(ConvLReLU, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=(3, 3),
                                           stride=1,
                                           use_bias=use_bias)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor):
        tensor = self.conv(input_tensor)
        tensor = self.lrelu(tensor)
        return tensor


class ResidualDenseBlock(tf.keras.Model):
    """Residual dense block insided RRDB blocks

    This block defines a Residual dense block which is used inside
    a RRDB block. A residual dense block consists of 5 convolutions
    seperated by leaky ReLU convolutions.The inputs of every
    convolution are propagated to the succeding convolutions and are
    concatenated with the inputs execept for the second last layer.

    Arguments:
    growth_channels: number of intermediate channels
    filters: number of output filters in each convolution
    use_bias: refer tf.keras.layers.Conv2D
    """
    def __init__(self, growth_channels=32, filters=64, use_bias=True):
        super(ResidualDenseBlock, self).__init__()
        self.input_convlr = ConvLReLU(filters=growth_channels,
                                      use_bias=use_bias)

        self.inner_layers = [ConvLReLU(filters=growth_channels,
                                       use_bias=use_bias)
                             for i in range(3)]

        self.output_conv = tf.keras.layers.Conv2D(filters=filters,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  use_bias=use_bias)

    def call(self, input_tensor):
        residue = [input_tensor]
        residue.append(self.input_convlr(input_tensor))

        for layer in self.inner_layers:
            tensor = layer(tf.concat(residue, 1))
            residue.append(tensor)

        return self.output_conv(tf.concat(residue, 1))


class RRDB(tf.keras.Model):
    """Residual in Residual Dense Block

    Constructs a residual in residual dense block from three
    ResidualDenseBlock instances. The residual dense blocks are
    stacked sequentially and are scaled up by a factor of beta.
    Now each the sum of the input tensors and the output of the
    previous units are fed into each block. The final output is
    again scaled up by a factor of beta.

    Arguments:
    beta: hyperparameter for RRDB block
    growth_channels, filters, use_bias: refer ResidualDenseBlock
    """
    def __init__(self, beta=0.2, growth_channels=32,
                 filters=64, use_bias=True):
        super(RRDB, self).__init__()
        self.dense_blocks = [ResidualDenseBlock(growth_channels,
                                                filters,
                                                use_bias)
                             for i in range(3)]
        self.beta = beta

    def call(self, input_tensor):
        tensor = input_tensor
        for block in self.dense_blocks:
            tensor = input_tensor + self.beta*block(tensor)

        return tensor
