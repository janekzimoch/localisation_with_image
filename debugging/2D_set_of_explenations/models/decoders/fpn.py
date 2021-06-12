from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from models.encoders.resnet_18 import *

def resnet_fpn(input_shape, num_heads=1):

    model = fpn(resnet_18, input_shape, num_heads=num_heads)
    model.model_name = f"resnet_fpn__{num_heads}_heads"
    
    return model


def fpn(encoder, input_shape, num_heads=1):

    image_input = layers.Input(shape=input_shape)
    # mask = layers.Input(shape=input_shape)
    
    image_input, levels = encoder(image_input)
    [f0, f1,f2,f3,f4] = levels
    # paper says: f0 should be disregarded due to high memory footprint.

    # ***** FPN *****
    # 7x7 to 14x14
    P4 = f4
    f4_prime = layers.Conv2D(256, (1,1))(P4)
    x = layers.UpSampling2D((2, 2))(f4_prime)
    f3_prime = layers.Conv2D(256, (1,1))(f3)
    x = layers.Add()([x, f3_prime])
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    P3 = layers.Activation("relu")(x)


    # 14x14 to 28x28
    x = layers.UpSampling2D((2, 2))(P3)
    f2_prime = layers.Conv2D(256, (1,1))(f2)
    x = layers.Add()([x, f2_prime])
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    P2 = layers.Activation("relu")(x)


    # 28x28 to 56x56
    x = layers.UpSampling2D((2, 2))(P2)
    f1_prime = layers.Conv2D(256, (1,1))(f1)
    x = layers.Add()([x, f1_prime])
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    P1 = layers.Activation("relu")(x)



    # ***** UPSAMPLE *****
    # upsample to 1/4 org. image
    # P4 - 1
    up_p4 = layers.Conv2D(128, (3, 3), padding='same')(P4)
    up_p4 = layers.BatchNormalization(axis=-1)(up_p4)  # tfa.layers.GroupNormalization()
    up_p4 = layers.Activation("relu")(up_p4)
    up_p4 = layers.UpSampling2D((2, 2), interpolation="bilinear")(up_p4)

    # P4 - 2
    up_p4 = layers.Conv2D(128, (3, 3), padding='same')(up_p4)
    up_p4 = layers.BatchNormalization(axis=-1)(up_p4)  # tfa.layers.GroupNormalization()
    up_p4 = layers.Activation("relu")(up_p4)
    up_p4 = layers.UpSampling2D((2, 2), interpolation="bilinear")(up_p4)

    # P4 - 3
    up_p4 = layers.Conv2D(128, (3, 3), padding='same')(up_p4)
    up_p4 = layers.BatchNormalization(axis=-1)(up_p4)  # tfa.layers.GroupNormalization()
    up_p4 = layers.Activation("relu")(up_p4)
    up_p4 = layers.UpSampling2D((2, 2), interpolation="bilinear")(up_p4)



    # P3 - 1
    up_p3 = layers.Conv2D(128, (3, 3), padding='same')(P3)
    up_p3 = layers.BatchNormalization(axis=-1)(up_p3)  # tfa.layers.GroupNormalization()
    up_p3 = layers.Activation("relu")(up_p3)
    up_p3 = layers.UpSampling2D((2, 2), interpolation="bilinear")(up_p3)

    # P3 - 2
    up_p3 = layers.Conv2D(128, (3, 3), padding='same')(up_p3)
    up_p3 = layers.BatchNormalization(axis=-1)(up_p3)  # tfa.layers.GroupNormalization()
    up_p3 = layers.Activation("relu")(up_p3)
    up_p3 = layers.UpSampling2D((2, 2), interpolation="bilinear")(up_p3)



    # P2 - 1
    up_p2 = layers.Conv2D(128, (3, 3), padding='same')(P2)
    up_p2 = layers.BatchNormalization(axis=-1)(up_p2)  # tfa.layers.GroupNormalization()
    up_p2 = layers.Activation("relu")(up_p2)
    up_p2 = layers.UpSampling2D((2, 2), interpolation="bilinear")(up_p2)

    # P2
    up_p1 = layers.Conv2D(128, (3, 3), padding='same')(P1)
    up_p1 = layers.BatchNormalization(axis=-1)(up_p1)  # tfa.layers.GroupNormalization()
    up_p1 = layers.Activation("relu")(up_p1)


    # SUM ELEMENT WISE
    y = layers.Add()([up_p4, up_p3, up_p2, up_p1])
    y = layers.BatchNormalization(axis=-1)(y)  # tfa.layers.GroupNormalization()
    y = layers.UpSampling2D((4, 4), interpolation="bilinear")(y)
    y = layers.Conv2D(64, (3, 3), padding='same')(y)

    y = layers.BatchNormalization(axis=-1)(y)  # tfa.layers.GroupNormalization()
    y = layers.Activation("relu")(y)



    # REGRESSION
    stacked_many_heads_output = None
    for i in range(num_heads):  # get multiple heads working
        output_reg = layers.Conv2D(3, (3, 3), padding='same')(y)
        # output_masked_reg = layers.Multiply()([output_reg, mask])
        
        if stacked_many_heads_output == None:
            stacked_many_heads_output = output_reg
        else:
            stacked_many_heads_output = layers.Concatenate(axis=-1)([stacked_many_heads_output, output_reg])
    

    # model = Model(inputs=[image_input, mask], outputs=stacked_many_heads_output) 
    model = Model(inputs=image_input, outputs=stacked_many_heads_output) 

#     model.summary()
    
    return model
