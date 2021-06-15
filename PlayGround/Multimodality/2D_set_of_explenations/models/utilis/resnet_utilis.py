import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.regularizers import l2
from keras.layers.merge import add



def conv_bn_relu(**conv_params):
    " builds: conv -> BN -> relu block "
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer)(input)

        norm = layers.BatchNormalization(axis=-1)(conv)
        act = layers.Activation("relu")(norm)
        return act

    return f


def bn_relu_conv(**conv_params):
    """
    same as - conv_bn_relu - but with changed order
    builds: BN -> relu block ->  conv 
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    # kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        norm = layers.BatchNormalization(axis=-1)(input)
        act = layers.Activation("relu")(norm)
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding=padding,
                        kernel_initializer=kernel_initializer, #kernel_regularizer=kernel_regularizer
                        )(act)
        return conv

    return f



def basic_block(filters, init_strides=(1, 1)):

    def f(input):
        conv = bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                strides=init_strides)(input)

        residual = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv)
        
        shortcut = input
        if init_strides == (2,2):
            shortcut = layers.Conv2D(filters=filters, kernel_size=(1, 1),
                          strides=init_strides,
                          padding="valid",
                          kernel_initializer="he_normal")(shortcut)
        output = add([shortcut, residual])
        return output

    return f



def basic_block_first(filters, init_strides=(1, 1)):

    def f(input):
        conv = layers.Conv2D(filters=filters, kernel_size=(3, 3),
                        strides=init_strides,
                        padding="same",
                        kernel_initializer="he_normal")(input)

        residual = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv)
        
        shortcut = input
        if init_strides == (2,2):
            shortcut = layers.Conv2D(filters=filters, kernel_size=(1, 1),
                          strides=init_strides,
                          padding="valid",
                          kernel_initializer="he_normal")(shortcut)
        output = add([shortcut, residual])
        return output

    return f