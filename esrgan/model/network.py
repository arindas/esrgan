"""
Module to implement the generator and discriminator network
architectures as mentioned in the ESRGAN paper.
Reference: https://arxiv.org/pdf/1809.00219.pdf
"""
import tensorflow as tf
from esrgan.model.blocks import (
    RRDB,
    ConvLReLU,
    UpSamplingBlock
)


class Generator(tf.keras.Model):  # pylint: disable=too-few-public-methods
    """ESRGAN Generator network composed of RRDB blocks.

    tf.keras.Model implementation of the ESRGAN generator network.
    An LR image is fed into a convolution followed by a series of a
    RRDB blocks, which is further followed by a convolution. The
    output of the first convolution is concatenated with the output
    of the convolution after the RRDB blocks as a residual connection
    and fed into an upsampling block. This followed by a pair of
    convolutions sequentially. The final output is the HR image.

    Builds an RRDBNetwork from the given parameters.

    Arguments:
    blocks: number of basic (RRDB) blocks to be used
    filters: number of output channels in convolutions outside of RRDBs
    growth_channels: intermediate output channels in residual dense blocks
    """
    def __init__(self,
                 blocks=5,
                 filters=64,
                 growth_channels=32):
        super(Generator, self).__init__()

        self.input_conv = tf.keras.layers.Conv2D(filters=filters,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding='same')

        self.rrdb_trunk = tf.keras.Sequential(
            [RRDB(filters=filters,
                  growth_channels=growth_channels)
             for i in range(blocks)])

        self.trunk_conv = tf.keras.layers.Conv2D(filters=filters,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding='same')

        self.upsample = tf.keras.Sequential(
            [UpSamplingBlock(filters=filters)
             for i in range(2)])

        self.hr_conv = tf.keras.layers.Conv2D(filters=filters,
                                              kernel_size=(3, 3),
                                              strides=1,
                                              padding='same')
        self.output_conv = tf.keras.layers.Conv2D(filters=3,
                                                  kernel_size=(3, 3),
                                                  strides=1,
                                                  padding='same')

    def call(self, input_tensor):
        """Foward propagation on input_tensor.

        Arguments:
        input_tensor: tensor to operate on
        """
        single_conv = self.input_conv(input_tensor)
        trunk = self.trunk_conv(self.rrdb_trunk(single_conv))
        trunk = single_conv + trunk

        upsampled_tensor = self.upsample(trunk)
        hr_tensor = self.hr_conv(upsampled_tensor)
        return self.output_conv(hr_tensor)


def get_discriminator_trunk():
    """Builds and returns the ESRGAN discriminator trunk"""
    layers = [
        ConvLReLU(),
        ConvLReLU(use_batch_norm=True, strides=2),
        ConvLReLU(filters=128, use_batch_norm=True, strides=1),
        ConvLReLU(filters=128, use_batch_norm=True, strides=2),
        ConvLReLU(filters=256, use_batch_norm=True, strides=1),
        ConvLReLU(filters=256, use_batch_norm=True, strides=2),
        ConvLReLU(filters=512, use_batch_norm=True, strides=1),
        ConvLReLU(filters=512, use_batch_norm=True, strides=2),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(1)
    ]
    return tf.keras.Sequential(layers)
