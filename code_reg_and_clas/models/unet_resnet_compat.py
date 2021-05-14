from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from models.encoders.resNet_18_encoder import *

from tensorflow.keras.models import Model

def vgg_unet(num_regions, input_height=224, input_width=224):

    print('Running: ResNet 18 backbone')
    model = unet(resNet_18_encoder, num_regions, input_height=input_height, input_width=input_width)
    model.model_name = "vgg_unet"
    
    return model


def unet(encoder, num_regions, input_height=224,
          input_width=224):

    image_input = layers.Input(shape=(224, 224, 3))
    mask = layers.Input(shape=(224, 224, 3))
    
    image_input, levels = encoder(image_input)
    [f0, f1, f2, f3, f4] = levels

    o = f4  # 7x7
    o = (layers.ZeroPadding2D((1, 1), name='U_block1_zero_pad'))(o)
    o = (layers.Conv2D(512, (3, 3), padding='valid' , activation='relu', name='U_block1_conv'))(o)
    o = (layers.BatchNormalization(name='U_block1_batch_norm'))(o)

    o = (layers.UpSampling2D((2, 2), name='U_block2_up_sample'))(o)  # output: 14x14
    o = (layers.concatenate([o, f3], axis=-1, name='U_block2_concat'))
    o = (layers.ZeroPadding2D((1, 1)))(o)
    o = (layers.Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (layers.BatchNormalization())(o)

    o = (layers.UpSampling2D((2, 2)))(o)  # output: 28x28
    o = (layers.concatenate([o, f2], axis=-1))
    o = (layers.ZeroPadding2D((1, 1)))(o)
    o = (layers.Conv2D(256, (3, 3), padding='valid' , activation='relu'))(o)
    o = (layers.BatchNormalization())(o)

    o = (layers.UpSampling2D((2, 2)))(o)  # ouptut: 56x56
    o = (layers.concatenate([o, f1], axis=-1))
    o = (layers.ZeroPadding2D((1, 1)))(o)
    o = (layers.Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (layers.BatchNormalization())(o)

    o = (layers.UpSampling2D((2, 2)))(o)  # ouptut: 112x112
    o = (layers.concatenate([o, f0], axis=-1))
    o = (layers.ZeroPadding2D((1, 1)))(o)
    o = (layers.Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (layers.BatchNormalization())(o)

    o = layers.Conv2D(128, (3, 3), padding='same')(o)
    o = (layers.UpSampling2D((2, 2)))(o) # output: 224x224
    o = (layers.ZeroPadding2D((1, 1)))(o)
    o = (layers.Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (layers.BatchNormalization())(o)
    

    # REGRESSION
    output_reg = layers.Conv2D(3, (3, 3), padding='same')(o)
    output_masked_reg = layers.Multiply()([output_reg, mask])

    
    # CLASSIFICATION
    o_clas = layers.Conv2D(num_regions, (3, 3), padding='same')(o)
    output_clas = keras.activations.softmax(o_clas, axis=-1)
    
    
    # CONCATENATE
    output = layers.Concatenate(axis=-1)([output_masked_reg, output_clas])
    
    
    model = Model(inputs=[image_input, mask], outputs=output) 
    # model.summary()
    
    return model
