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
# from models.unet import *
from models.unet_resnet_compat import *


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
    copyfile("run_experiment_r_c.py", experiment_full_name + "/code/run_experiment_r_c.py")
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

    if kwargs['tensorboard'] == True:
        logdir = kwargs['experiment_dir'] + "logs/" + kwargs['experiment_name']
        if not os.path.exists(logdir):
            os.makedirs(logdir)
            
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)

    if kwargs['pixelwise_MSE'] == True:
        file_name = "/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512_complete_working_141_classes/0001_rendered.png_config.npz"

        global_coord_mse_callback = pixelwise_MSE(file_name, 1, 20, kwargs['experiment_dir'], kwargs['experiment_name'])
        callbacks.append(global_coord_mse_callback)

    if kwargs['save_input_images'] == True:
        [images, mask], labels = train_gen.__getitem__(0)
        save_sample_input = Save_sample_input([images, mask], labels, kwargs['experiment_dir'], kwargs['experiment_name'])
        callbacks.append(save_sample_input)

    if kwargs['train_visualisation'] == True:
        [images, mask], labels = train_gen.__getitem__(0)
        vis_learning = Visualise_learning_reg_and_class(images[0], mask[0], labels[0,:,:,:3], labels[0,:,:,3:], 
                                            kwargs['vis_frequency'], kwargs['experiment_dir'], kwargs['experiment_name'], "train/")
        callbacks.append(vis_learning)

    if kwargs['val_visualisation'] == True:
        [images, mask], labels = val_gen.__getitem__(0)
        vis_learning = Visualise_learning_reg_and_class(images[0], mask[0], labels[0,:,:,:3], labels[0,:,:,3:],
                                            kwargs['vis_frequency'], kwargs['experiment_dir'], kwargs['experiment_name'], "val/")
        callbacks.append(vis_learning)

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


def combined_loss_weighted(weights):
    reg_w, clas_w = weights

    def combined_loss(y_true, y_pred):
        " combined loss for 3D local coordinate regression and Classification"

        regr_true = y_true[:,:,:,:3]
        clas_true = y_true[:,:,:,3:]    
        regr_pred = y_pred[:,:,:,:3]
        clas_pred = y_pred[:,:,:,3:]

        regr_loss = tf.keras.losses.mean_squared_error(regr_true, regr_pred)
        regr_loss = tf.reduce_mean(regr_loss)

    #     clas_loss = tf.keras.losses.categorical_crossentropy(clas_true, clas_pred)  # 1-hot-encoded labels
        clas_loss = tf.keras.losses.sparse_categorical_crossentropy(clas_true, clas_pred)  # categorical labels 
        clas_loss = tf.reduce_mean(clas_loss)
        
        loss = reg_w*regr_loss + clas_w*clas_loss
        return loss
        
    return combined_loss


def setup_model(compile_configs, **kwargs):
    
    if not kwargs['run_from_checkpoint']:
        unet_model = vgg_unet(num_regions=kwargs['num_regions'])
        print('reg weight: ', kwargs['weights'][0], '   clas weight: ', kwargs['weights'][1])
        unet_model.compile(optimizer=keras.optimizers.Adam(kwargs['learning_rate']), loss=combined_loss_weighted(kwargs['weights']))
    else:
        unet_model = keras.models.load_model(kwargs['checkpoint_dir'], compile=False)
        unet_model.compile(optimizer=keras.optimizers.Adam(kwargs['learning_rate']), loss=combined_loss_weighted(kwargs['weights']))
    return unet_model


def save_history(experiment_full_name, history):
    hist_json_file = experiment_full_name + '/train_history.json' 

    with open(hist_json_file, 'w') as history_file:
        json.dump(history, history_file, sort_keys=True, indent=4)
