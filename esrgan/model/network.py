"""
Module to implement the generator and discriminator network
architectures as mentioned in the ESRGAN paper.
Reference: https://arxiv.org/pdf/1809.00219.pdf
"""
import tensorflow as tf
from esrgan.model.blocks import (
    RRDB,
    UpSamplingBlock
)


class Generator(tf.keras.Model):
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
    filters:
    growth_channels:
    blocks: number of RRDB blocks to be used

    """
    def __init__(self,
                 blocks=5,
                 filters=64,
                 growth_channels=32):
        super(Generator, self).__init__()

        self.input_conv = tf.keras.layers.Conv2D(filters=filters,
                                                 kernel_size=(3, 3),
                                                 stride=1)

        self.rrdb_trunk = tf.keras.Sequential(
            [RRDB(filters=filters,
                  growth_channels=growth_channels)
             for i in range(blocks)])

        self.trunk_conv = tf.keras.layers.Conv2D(filters=filters,
                                                 kernel_size=(3, 3),
                                                 stride=1)

        self.upsample1 = UpSamplingBlock(filters=filters)
        self.upsample2 = UpSamplingBlock(filters=filters)

        self.hr_conv = tf.keras.layers.Conv2D(filters=filters,
                                              kernel_size=(3, 3),
                                              stride=1)
        self.output_conv = tf.keras.layers.Conv2D(filters=3,
                                                  kernel_size=(3, 3),
                                                  stride=1)

    def call(self, input_tensor):
        single_conv = self.input_conv(input_tensor)
        trunk = self.trunk_conv(self.rrdb_trunk(single_conv))
        trunk = single_conv + trunk

        upsampled_tensor = self.upsample2(self.upsample1(trunk))
        return self.output_conv(upsampled_tensor)


class Discriminator(tf.keras.Model):
    pass
