from tensorflow.keras import layers
from tensorflow.keras.models import Model


def basic_model(encoder, num_heads=1, output_dim=1, input_shape=(224,224,1)):
    
    image_input = layers.Input(shape=input_shape)
    image_input, levels = encoder(image_input)
    [f0,f1,f2,f3,f4] = levels
    
    x = f4
    head_input = layers.GlobalAveragePooling2D(name='pool1')(x)
    if num_heads > 1:
        heads = []
        for i in range(num_heads):
            h = layers.Dense(64, activation='relu', name=f'fc{i}_0')(head_input)
            output = layers.Dense(output_dim*num_heads, name=f'fc{i}')(h)
            heads.append(output)

        output = layers.Concatenate(axis=-1)(heads)
    else:
        h = layers.Dense(64, activation='relu', name='fc0')(head_input)
        output = layers.Dense(output_dim*num_heads, name='fc1')(h)

    
    model = Model(inputs=image_input, outputs=output) 
    # model.summary()
    
    return model



def basic_localisation_model_disconected_graph(encoder, num_heads=1, input_shape=(224,224,1)):
    """
    This is the same model as the one from 'basic_localisation_model()', however we disconect gradients from 
    the extra heads from the trunk. Such that the extra heads only modify weights associated with their heads.
    Advantages of this solution:
    * the main head learn the best possible answer - ... this could actually be not desirable because 
    the main head could learn an MSE solution which doesn't conincide with any mode, but just lies in the middle.
    """
    image_input = layers.Input(shape=input_shape)
    image_input, levels = encoder(image_input)
    [f0,f1,f2,f3,f4] = levels
    
    x = f4
    head_input = layers.GlobalAveragePooling2D(name='pool1')(x)
    if num_heads > 1:
        heads = []
        for i in range(num_heads):
            if i == 0:
                h = head_input
            else:
                h = tf.stop_gradient(head_input)
                
            h = layers.Dense(64, activation='relu', name=f'fc{i}_0')(h)
            output = layers.Dense(3, name=f'fc{i}')(h)
            heads.append(output)

        output = layers.Concatenate(axis=-1)(heads)
    else:
        x = layers.Dense(64, activation='relu', name='fc0')(x)
        output = layers.Dense(3, name='fc1')(x)

    
    model = Model(inputs=image_input, outputs=output) 
    
    return model