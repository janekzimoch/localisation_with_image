import numpy as np
from utilis.pose__get_coord_predictions import *
from pose.get_pose import *
import random
random.seed(10)

def get_groundtruth_data(datapoint_file_name):
    
    data_dir = "/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512/"
    npz_data = np.load(data_dir + datapoint_file_name)
        
    pixel_bearing = npz_data['points_3d_sphere']
    pose = {'T': npz_data['T_blender'], 'R': npz_data['R_blender']}
    
    return pixel_bearing, pose


def filter_with_mask(coords, masks):
    masks_flat = np.reshape(masks, (-1)).astype(bool)
    coords_flat = np.reshape(coords, (-1,3))
    
    return coords_flat[masks_flat,:]


def get_pose_errors(model_dir, simple_whitenning=False):
    
    # get dataset and predictions
    dataset = load_validation_images(model_dir, simple_whitenning=simple_whitenning)
    model = load_model(model_dir)
    print('predicting...')
    pred_local_coords = predict(model, dataset['images'], dataset['masks'])
    pred_global_coords, pred_regions = unwhitten(pred_local_coords, dataset, simple_whitenning=simple_whitenning)
    pred_global_coords = np.reshape(pred_global_coords, (len(dataset['images']),256,512,3))
    
    # get validation file names
    data_partition = get_data_partition(model_dir)
    dataset_size = len(data_partition['validation'])
    
    localisation_errors = np.zeros(dataset_size)
    orientation_errors = np.zeros(dataset_size)
    
    for i, datapoint in tqdm(enumerate(data_partition['validation'])):
        datapoint_file_name = datapoint.split('/')[-1]

        # need to get GT - T and R matrices for each val image
        bearing, gt_pose = get_groundtruth_data(datapoint_file_name)
        bearing_filtered = filter_with_mask(bearing, dataset['masks'][i])
        
        # need to predict T and R matrices for each val image
        pred_global_coord_flat = np.reshape(pred_global_coords[i], (-1,3))
        pred_global_coords_filtered = fil   ter_with_mask(pred_global_coord_flat, dataset['masks'][i])
        
        num_points = len(pred_global_coords_filtered)
        indexes = random.sample(range(num_points), int(num_points/10))
        pred_pose = get_pose_all_points(pred_global_coords_filtered[indexes], bearing_filtered[indexes], threshold=0.004)
        
        # get pose errors
        orient_error, loc_error = get_pose_error(pred_pose, gt_pose)
        
        localisation_errors[i] = loc_error
        orientation_errors[i] = orient_error
        
    return localisation_errors, orientation_errors
