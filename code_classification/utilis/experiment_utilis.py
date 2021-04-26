import os, sys
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import datetime
import json
from shutil import copyfile, copytree

import tensorflow as tf
from tensorflow import keras


from utilis.data_generator import *
from utilis.callbacks import *
from models.unet import *
from models.encoders.vgg_encoder import *


def get_configs(json_configs_file):
    configs_file = open(json_configs_file)
    configs = json.load(configs_file)
    configs_file.close()

    return configs



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

    # copy configs.json
    if not os.path.exists(experiment_full_name + "/code"):
        os.makedirs(experiment_full_name + "/code")
    copyfile("configs.json", experiment_full_name+"/code/configs.json")

    # copy code
    copyfile("run_experiment_clas.py", experiment_full_name + "/code/run_experiment_clas.py")
    copytree('models', experiment_full_name + "/code" + "/models")
    copytree('utilis', experiment_full_name + "/code" + "/utilis")

    return experiment_name, experiment_full_name


def get_data_generator(data_partition, generator_configs):
    training_generator = DataGenerator(data_partition['train'], **generator_configs)
    validation_generator = DataGenerator(data_partition['validation'], **generator_configs)

    return training_generator, validation_generator


def get_data(experiment_full_name, dataset_size,
             data_dir,
             val_split,
             **kwargs):
  
    if kwargs['run_from_checkpoint'] == True:
        # get data_partition file from the previous checkpoint's experiment directory
        data_partition_dir = "/".join(kwargs['checkpoint_dir'].split('/')[:-1]) + '/data_partition.json'
        data_partition_file = open(data_partition_dir)
        data_partition = json.load(data_partition_file)
        data_partition_file.close()

    else:
        data_partition = {'train': [],
                        'validation': []}

        indexes = np.arange(1,dataset_size + 1)
        np.random.shuffle(indexes)
        split_index = int(val_split*dataset_size)

        for ind in indexes[split_index:]:
            coord_npz = f"{ind:04}_rendered.png_config.npz"
            data_partition['train'].append(data_dir + coord_npz)

        for ind in indexes[:split_index]:
            coord_npz = f"{ind:04}_rendered.png_config.npz"
            data_partition['validation'].append(data_dir + coord_npz)


    # save data_partition file to your experiment directory
    with open(experiment_full_name + '/data_partition.json', 'w') as json_file:
        json.dump(data_partition, json_file, sort_keys=True, indent=4)
    
    return data_partition



def get_callbacks(train_gen, val_gen,**kwargs):
    callbacks = []

    if kwargs['save_input_images'] == True:
        images, labels = train_gen.__getitem__(0)
        save_sample_input = Save_sample_input(images, labels, kwargs['experiment_name'])
        callbacks.append(save_sample_input)

    if kwargs['train_visualisation'] == True:
        images, labels = train_gen.__getitem__(0)
        vis_learning = Visualise_learning(images[0], labels[0], 
                                            kwargs['vis_frequency'], kwargs['experiment_name'], "train/")
        callbacks.append(vis_learning)

    if kwargs['val_visualisation'] == True:
        images, labels = val_gen.__getitem__(0)
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
            mode='auto',
            period=10 #save_freq=10*33 #
            )
        callbacks.append(modelCheckpoint_callback)
  
    if kwargs['garbage_cleaner'] == True:
        callbacks.append(RemoveGarbageCallback())

    return callbacks


def masked_MSE(y_true, y_pred):
    mask = y_true[:,:,:,3]
    y_true = y_true[:,:,:,:3]

    mask_expanded = tf.stack([mask,mask,mask], axis=-1)

    y_pred = tf.math.multiply(y_pred, mask_expanded)
    y_true = tf.math.multiply(y_true, mask_expanded)


    squared_difference = tf.square(y_true - y_pred)
    # loss = tf.reduce_mean(squared_difference, axis=-1)
    loss = tf.reduce_mean(squared_difference)

    return loss

def masked_X_entropy(y_true, y_pred):
    " masked CrossEntropy - used for classification of regions "

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(loss)



def setup_model(compile_configs, **kwargs):
    
    if not kwargs['run_from_checkpoint']:
        unet_model = vgg_unet()
        unet_model.compile(optimizer=keras.optimizers.Adam(kwargs['learning_rate']), loss=masked_X_entropy)
    else:
        unet_model = keras.models.load_model(kwargs['checkpoint_dir'], compile=False)
        unet_model.compile(optimizer=keras.optimizers.Adam(kwargs['learning_rate']), loss=masked_X_entropy)
    return unet_model


def save_history(experiment_full_name, history):
    hist_json_file = experiment_full_name + '/train_history.json' 

    with open(hist_json_file, 'w') as history_file:
        json.dump(history, history_file, sort_keys=True, indent=4)
