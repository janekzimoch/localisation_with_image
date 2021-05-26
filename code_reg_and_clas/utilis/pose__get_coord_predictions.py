import numpy as np
import json
from tqdm import tqdm

from utilis.experiment_utilis import combined_loss_weighted
import tensorflow as tf
from tensorflow import keras 

def get_data_partition(model_dir):
    # load json with data split
    data_partition_path = model_dir + "/data_partition.json"

    data_partition_file = open(data_partition_path)
    data_partition = json.load(data_partition_file)
    data_partition_file.close()
    
    return data_partition


def load_validation_images(model_dir, simple_whitenning=False):
    
    data_partition = get_data_partition(model_dir)
    dataset_size = len(data_partition['validation'])
    
    images = np.zeros((dataset_size, 256, 512, 3))
    oracle_global_coords = np.zeros((dataset_size, 256, 512, 3))
    regions = np.zeros((dataset_size, 256, 512, 1))
    local_coords = np.zeros((dataset_size, 256, 512, 3))
    masks = np.zeros((dataset_size, 256, 512, 1))

    dataset = {}
    for i, data_dir in enumerate(data_partition['validation']):
        data = np.load(data_dir)
        

        images[i] = data['image_colors']
        oracle_global_coords[i] = data['points_3d_world']
        regions[i,:,:,0] = data['points_region_class']
        local_coords[i] = data['local_scene_coords']
        masks[i,:,:,0] = data['mask']
        
        if i == 0:
            if not simple_whitenning:
                dataset['W_inv'] = data['W_inv']
                dataset['M'] = data['M']
                dataset['std'] = data['std']
            else:
                dataset['M'] = data['mean_vec']
                dataset['std'] = data['std_vec']
                
    dataset['images'] = images
    dataset['oracle_global_coords'] = oracle_global_coords
    dataset['regions'] = regions
    dataset['local_coords'] = local_coords
    dataset['masks'] = masks
    
    return dataset

def load_model(model_dir):
    # load model
    checkpoint_dir = model_dir + "/saved_model/my_model"
    model = keras.models.load_model(checkpoint_dir, compile=False)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=combined_loss_weighted([10, 1]))
    
    return model

def predict(model, images, masks):
    dataset_size = len(images)
    pred_local_coords = np.zeros((dataset_size, 256, 512, 144))
    for i in tqdm(range(dataset_size)):
        pred_local_coords[i] = np.array(model.predict_on_batch([images[i:i+1], masks[i:i+1]]))
    
    return pred_local_coords

def unwhitten(pred_local_coords, dataset, simple_whitenning=False):
    dataset_size = len(pred_local_coords)
    
    # 1. convert regions to their mean centers
    pred_local_coords_flat = np.reshape(pred_local_coords[:,:,:,:3], (-1,3))
    pred_regions = np.argmax(pred_local_coords[:,:,:,3:], axis=-1)
    pred_regions_flat = np.reshape(pred_regions, (-1)).astype(int)
    pred_global_coords = np.zeros((dataset_size*256*512,3))

    # 2. unwhitten local coordinates
    for region in tqdm(np.unique(pred_regions_flat)):
        region_coords = pred_local_coords_flat[pred_regions_flat == region]
        
        if not simple_whitenning:
            unwhite_loc_coords = np.dot(region_coords * dataset['std'][region] , dataset['W_inv'][region]) + dataset['M'][region]
        else:
            unwhite_loc_coords = (region_coords * dataset['std'][region]) + dataset['M'][region]

        pred_global_coords[pred_regions_flat == region] = unwhite_loc_coords

    pred_global_coords = np.reshape(pred_global_coords, (dataset_size, 256, 512, 3))
    
    return pred_global_coords, pred_regions



def get_sorted_MSE__GT_mask(pred_global_coords, oracle_global_coords, masks):
    pixelwise_MSE = np.mean(np.square((pred_global_coords*masks) - (oracle_global_coords*masks)), axis=-1)
    pixelwise_MSE = np.reshape(pixelwise_MSE, (-1))
    return np.sort(pixelwise_MSE)

def get_sorted_MSE__Pred_mask(pred_global_coords, oracle_global_coords, pred_regions):
    # get mask
    masks = np.where(pred_regions == 140, 0, 1)
    
    # get MSE
    pixelwise_MSE = np.mean(np.square((pred_global_coords*masks) - (oracle_global_coords*masks)), axis=-1)
    pixelwise_MSE = np.reshape(pixelwise_MSE, (-1))
    return np.sort(pixelwise_MSE)


