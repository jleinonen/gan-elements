import numpy as np
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import Input, Flatten, LeakyReLU, Reshape
from tensorflow.keras.layers import Concatenate, UpSampling2D, Multiply
from tensorflow.keras.models import Model


def cgan_disc(img_shape=(32,32,1)):

    def conv_block(channels, strides=2):
        def block(x):
            x = Conv2D(channels, kernel_size=3, strides=strides,
                padding="same")(x)
            x = LeakyReLU(0.2)(x)
            return x
        return block

    image_in = Input(shape=img_shape, name="sample_in")
    cond_in = Input(shape=(10,), name="cond_in")

    x = conv_block(64, strides=1)(image_in)
    x = conv_block(128)(x)
    x = conv_block(256)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    
    c = Dense(256)(cond_in)
    c = LeakyReLU(0.2)(c)
    c = Dense(256)(c)

    x = Multiply()([x,c])
    
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)

    disc_out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[image_in,cond_in], outputs=disc_out)

    return model


def cgan_gen(img_shape=(32,32,1), noise_dim=64):

    def up_block(channels):
        def block(x):
            x = UpSampling2D()(x)
            x = Conv2D(channels, kernel_size=3, padding="same")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(0.2)(x)
            return x
        return block

    cond_in = Input(shape=(10,), name="cond_in")
    noise_in = Input(shape=(noise_dim,), name="noise_in")
    inputs = Concatenate()([cond_in,noise_in])
    
    initial_shape = (img_shape[0]//4, img_shape[1]//4, 256)

    x = Dense(np.prod(initial_shape))(inputs)
    x = LeakyReLU(0.2)(x)
    x = Reshape(initial_shape)(x)
    x = up_block(256)(x)
    x = up_block(128)(x)
    img_out = Conv2D(img_shape[-1], kernel_size=3, padding="same", 
        activation="tanh")(x)

    return Model(inputs=[cond_in,noise_in], outputs=img_out)
