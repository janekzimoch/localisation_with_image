from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from models.encoders.resNet_18_encoder import *

from tensorflow.keras.models import Model

def resnet_fpn(num_regions, input_height=224, input_width=224):

    print('Running: VGG-16 backbone')
    model = fpn(resNet_18_encoder, num_regions, input_height=input_height, input_width=input_width)
    model.model_name = "resnet_fpn"
    
    return model


def fpn(encoder, num_regions, l1_skip_conn=True, input_height=224,
          input_width=224):

    image_input = layers.Input(shape=(224, 224, 3))
    mask = layers.Input(shape=(224, 224, 3))
    
    image_input, levels = encoder(image_input)
    [f0, f1,f2,f3,f4] = levels
    # paper says: f0 should be disregarded due to high memory footprint.

    # general strategy
    # 1. upsample a coarser but sematically stronger layer
    # 2. add a higher resolution feature mat from bottom-up network (via lateral connections
    
    # 7x7 to 14x14
    f4_prime = layers.Conv2D(256, (1,1))(f4)
    x = layers.UpSampling2D((2, 2))(f4_prime)
    f3_prime = layers.Conv2D(256, (1,1))(f3)
    x = layers.Add()([x, f3_prime])
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)


    # 14x14 to 28x28
    x = layers.UpSampling2D((2, 2))(x)
    f2_prime = layers.Conv2D(256, (1,1))(f2)
    x = layers.Add()([x, f2_prime])
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)


    # 28x28 to 56x56
    x = layers.UpSampling2D((2, 2))(x)
    f1_prime = layers.Conv2D(256, (1,1))(f1)
    x = layers.Add()([x, f1_prime])
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)


    # 56x56 to 112x112
    x = layers.UpSampling2D((2, 2))(x)
    f0_prime = layers.Conv2D(256, (1,1))(f0)
    x = layers.Add()([x, f0_prime])
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)


    # 112x112 to 224x224
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    # x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Activation("relu")(x)

  
    # REGRESSION
    output_reg = layers.Conv2D(3, (3, 3), padding='same')(x)
    output_masked_reg = layers.Multiply()([output_reg, mask])

    
    # CLASSIFICATION
    x_clas = layers.Conv2D(num_regions, (3, 3), padding='same')(x)
    output_clas = keras.activations.softmax(x_clas, axis=-1)
    
    
    # CONCATENATE
    output = layers.Concatenate(axis=-1)([output_masked_reg, output_clas])
    
    
    model = Model(inputs=[image_input, mask], outputs=output) 
#     model.summary()
    
    return model
