import os, sys
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import datetime

from tensorflow import keras

from utilis.data_generator import *
from utilis.callbacks import *
from models.unet import *
from models.encoders.vgg_encoder import *




def setup_experiment_folder(**kwargs):
    current_time = str(datetime.datetime.now()).split('.')[0]
    current_day = current_time.split(" ")[0] + "/"
    current_time = current_time.replace(' ', '_')

    # make experiment name
    experiment_name = current_day + kwargs['experiment_name'] + "_{}".format(current_time)
    experiment_full_name = kwargs['experiment_dir'] + experiment_name

    # make directory
    if not os.path.exists(experiment_full_name):
        os.makedirs(experiment_full_name)

    """ TODO: 
    1. Create directories: visualisation, modelcheckpoint, train_history_logs
    2. Copy over configs.json file
    3. copy code (need to decide how much code to copy. To start with just unet model and encoder could be fine)
    """

    return experiment_name


def get_data_generator(data_partition, generator_configs):
    training_generator = DataGenerator(data_partition['train'], **generator_configs)
    validation_generator = DataGenerator(data_partition['validation'], **generator_configs)

    return training_generator, validation_generator


def get_data(dataset_size=370,
             data_dir= "/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512/",
             val_split=0.05):
  
    data_partition = {'train': [],
                     'validation': []}

    indexes = np.arange(1,dataset_size + 1)
    np.random.shuffle(indexes)
    split_index = int(val_split*dataset_size)

    for ind in indexes[split_index:]:
        coord_npz = f"{ind:03}_rendered.png_config.npz"
        data_partition['train'].append(data_dir + coord_npz)

    for ind in indexes[:split_index]:
        coord_npz = f"{ind:03}_rendered.png_config.npz"
        data_partition['validation'].append(data_dir + coord_npz)
    
    return data_partition



def get_callbacks(train_gen, val_gen,**kwargs):
    callbacks = []

    if kwargs['train_visualisation'] == True:
        images, labels = train_gen.__getitem__(0)
        vis_learning = Visualise_learning(images[0], labels[0], 
                                            kwargs['vis_frequency'], kwargs['experiment_name'], "train/")
        callbacks.append(vis_learning)

    if kwargs['val_visualisation'] == True:
        images, labels = train_gen.__getitem__(0)
        vis_learning = Visualise_learning(images[0], labels[0], 
                                            kwargs['vis_frequency'], kwargs['experiment_name'], "val/")
        callbacks.append(vis_learning)

    if kwargs['tensorboard'] == True:
        logdir = kwargs['experiment_dir'] + "logs/" + kwargs['experiment_name']
        if not os.path.exists(logdir):
            os.makedirs(logdir)
            
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)

    if kwargs['model_checkpoint'] == True:
        modelCheckpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath= kwargs['experiment_dir'] + kwargs['experiment_name'] + "/model_checkpoint", 
            monitor='val_loss', 
            verbose=0, 
            save_best_only=True,
            mode='auto')
        callbacks.append(modelCheckpoint_callback)
        
    if kwargs['garbage_cleaner'] == True:
        callbacks.append(RemoveGarbageCallback())
        
    return callbacks



def setup_model(compile_configs, **kwargs):
    unet_model = vgg_unet()
    unet_model.compile(optimizer=keras.optimizers.Adam(kwargs['learning_rate']), **compile_configs)

    return unet_model