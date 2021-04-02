from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def unet(encoder, l1_skip_conn=True, input_height=224,
          input_width=224):

    img_input, levels = encoder(input_shape=(input_height, input_width, 3))
    [f1, f2, f3, f4, f5] = levels

    o = f4
    o = (layers.ZeroPadding2D((1, 1), name='U_block1_zero_pad'))(o)
    o = (layers.Conv2D(512, (3, 3), padding='valid' , activation='relu', name='U_block1_conv'))(o)
    o = (layers.BatchNormalization(name='U_block1_batch_norm'))(o)

    o = (layers.UpSampling2D((2, 2), name='U_block2_up_sample'))(o)
    o = (layers.concatenate([o, f3], axis=-1, name='U_block2_concat'))
    o = (layers.ZeroPadding2D((1, 1)))(o)
    o = (layers.Conv2D(512, (3, 3), padding='valid', activation='relu'))(o)
    o = (layers.BatchNormalization())(o)

    o = (layers.UpSampling2D((2, 2)))(o)
    o = (layers.concatenate([o, f2], axis=-1))
    o = (layers.ZeroPadding2D((1, 1)))(o)
    o = (layers.Conv2D(256, (3, 3), padding='valid' , activation='relu'))(o)
    o = (layers.BatchNormalization())(o)

    o = (layers.UpSampling2D((2, 2)))(o)

    if l1_skip_conn:
        o = (layers.concatenate([o, f1], axis=-1))

    o = (layers.ZeroPadding2D((1, 1)))(o)
    o = (layers.Conv2D(128, (3, 3), padding='valid', activation='relu'))(o)
    o = (layers.BatchNormalization())(o)

    o = layers.Conv2D(128, (3, 3), padding='same')(o)
    
    o = (layers.UpSampling2D((2, 2)))(o)
    o = (layers.ZeroPadding2D((1, 1)))(o)
    o = (layers.Conv2D(64, (3, 3), padding='valid', activation='relu'))(o)
    o = (layers.BatchNormalization())(o)
    
    o = layers.Conv2D(64, (3, 3), padding='same')(o)

    
    output = layers.Conv2D(3, (3, 3), padding='same')(o)
    
    model = Model(img_input, output) 
#     model.summary()
    
    return model