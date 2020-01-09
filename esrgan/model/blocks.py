"""
Module for defining all the blocks to be used in implementing an ESRGAN.
Reference: https://arxiv.org/pdf/1809.00219.pdf
"""

import tensorflow as tf


class ConvLReLU(tf.keras.Model):  # pylint: disable=too-few-public-methods
    """Conv LeakyReLU pair.

    Elementary block inside ResidualDenseBlock. It comprises of
    a 2D Convolution followed by a leaky ReLu activation. Optionally
    there may be batch normalization layer in the middle.

    Arguments:
    filters: number of output channels
    strides: strides of convolutions
    batch_norm: whether to use batch normalization
    use_bias: refer tf.keras.layers.Conv2D
    """
    def __init__(self, filters=64, strides=1,
                 use_batch_norm=False, use_bias=True):
        super(ConvLReLU, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=(3, 3),
                                           strides=strides,
                                           use_bias=use_bias)

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization()

        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor):
        """Forward propagation for ConvLReLU

        Arguments:
        input_tensor: tensor to be operated on
        """
        tensor = self.conv(input_tensor)
        if self.use_batch_norm:
            tensor = self.batch_norm(tensor)
        tensor = self.lrelu(tensor)
        return tensor


# pylint: disable=too-few-public-methods
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
                                                  strides=1,
                                                  use_bias=use_bias)

    def call(self, input_tensor):
        """Forward propagation for ResidualDenseBlock

        Arguments:
        input_tensor: tensor to be operated on
        """
        residue = [input_tensor]
        residue.append(self.input_convlr(input_tensor))

        for layer in self.inner_layers:
            tensor = layer(tf.concat(residue, 1))
            residue.append(tensor)

        return self.output_conv(tf.concat(residue, 1))


class RRDB(tf.keras.Model):  # pylint: disable=too-few-public-methods
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
        """Forward propagation for ResidualDenseBlock

        Arguments:
        input_tensor: tensor to be operated on
        """
        tensor = input_tensor
        for block in self.dense_blocks:
            tensor = input_tensor + self.beta*block(tensor)

        return tensor


# pylint: disable=too-few-public-methods
class UpSamplingBlock(tf.keras.Model):
    """Up sampling block.

    Constructs an up sampling block from a 2D convolution, an
    upsampling layer(2x) and a LeakyReLU activation.

    Arguments:
    filters: number of output channels of the convolution
    """
    def __init__(self, filters=64, use_bias=True):
        super(UpSamplingBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=(3, 3),
                                           strides=1,
                                           use_bias=use_bias)
        self.upsample = tf.keras.layers.UpSampling2D(size=2)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, input_tensor):
        """Forward propagation for ResidualDenseBlock

        Arguments:
        input_tensor: tensor to be operated on
        """
        return self.lrelu(self.conv(self.upsample(input_tensor)))
