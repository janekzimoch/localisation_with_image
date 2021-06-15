import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def mdn_model(encoder, num_comp=1, output_shape=1, input_shape=(224,224,1)):
    
    image_input = layers.Input(shape=input_shape)
    image_input, levels = encoder(image_input)
    [f0,f1,f2,f3,f4] = levels

    x = f4
    x = layers.GlobalAveragePooling2D(name='pool1')(x)

    mix_components = layers.Dense(num_comp)(x)
    means = layers.Dense(output_shape*num_comp)(x)
    log_variances = layers.Dense(output_shape*num_comp)(x)
    variances = tf.exp(log_variances)

    output = layers.Concatenate()([mix_components, means, variances])
    model = Model(inputs=image_input, outputs=output)

    return model