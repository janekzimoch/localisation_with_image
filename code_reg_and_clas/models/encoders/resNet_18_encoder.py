import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.resnet_util import *

def resNet_18_encoder(input_shape=(224, 224, 3)):

    image_input = layers.Input(shape=(224, 224, 3))

    x = conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(image_input)  # output: 112x112
    f0 = x
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)  # output: 56x56

    # BLOCK 1 - 64 filters
    x = basic_block_first(64, init_strides=(1, 1))(x) # output: 56x56
    x = basic_block(64, init_strides=(1, 1))(x)
    f1 = x

    # BLOCK 2 - 128 filters
    x = basic_block(128, init_strides=(2, 2))(x)  # output: 28x28
    x = basic_block(128, init_strides=(1, 1))(x)
    f2 = x

    # BLOCK 3 - 256 filters
    x = basic_block(256, init_strides=(2, 2))(x)  # output: 14x14
    x = basic_block(256, init_strides=(1, 1))(x)
    f3 = x

    # BLOCK 4 - 512 filters
    x = basic_block(512, init_strides=(2, 2))(x) # output: 7x7
    x = basic_block(512, init_strides=(1, 1))(x)
    
    # FINAL ACTIVATION
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)
    f4 = x 


    return image_input, [f0, f1,f2,f3,f4]


