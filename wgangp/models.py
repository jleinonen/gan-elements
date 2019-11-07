import numpy as np
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import Input, Flatten, LeakyReLU, Reshape
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model


def dcgan_disc(img_shape=(32,32,1)):
    # Adapted from:
    # https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

    def conv_block(channels, strides=2):
        def block(x):
            x = Conv2D(channels, kernel_size=3, strides=strides,
                padding="same")(x)
            x = LeakyReLU(0.2)(x)
            return x
        return block

    image_in = Input(shape=img_shape, name="sample_in")

    x = conv_block(64, strides=1)(image_in)
    x = conv_block(128)(x)
    x = conv_block(256)(x)
    x = Flatten()(x)
    disc_out = Dense(1, activation="linear")(x)

    model = Model(inputs=image_in, outputs=disc_out)

    return model


def dcgan_gen(img_shape=(32,32,1), noise_dim=64):
    # Adapted from:
    # https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

    def up_block(channels):
        def block(x):
            x = UpSampling2D()(x)
            x = Conv2D(channels, kernel_size=3, padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(0.2)(x)
            return x
        return block

    noise_in = Input(shape=(noise_dim,), name="noise_in")
    initial_shape = (img_shape[0]//4, img_shape[1]//4, 256)

    x = Dense(np.prod(initial_shape))(noise_in)
    x = LeakyReLU(0.2)(x)
    x = Reshape(initial_shape)(x)
    x = up_block(128)(x)
    x = up_block(64)(x)
    img_out = Conv2D(img_shape[-1], kernel_size=3, padding="same", 
        activation="tanh")(x)

    return Model(inputs=noise_in, outputs=img_out)
