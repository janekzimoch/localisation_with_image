import json

configs = {
    'experiment_info': {
        'experiment_name': "FPN_resnet_bs16_LR5e-4_w1-1_simple_whitenning",
        'experiment_dir': '/data/cornucopia/jz522/experiments/',
        'multiple_GPU': False,
        'GPUs': ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']
    },
    'data_partition': {
        'dataset_size': 370,
        'data_dir': "/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512__141_classes__simple_whitening/",
        'val_split': 0.1
    },

    'callbacks': {
        'save_input_images': True,
        'pixelwise_MSE_single': False,
        'pixelwise_MSE_train': True,
        'pixelwise_MSE_val': True,
        'train_visualisation': True,
        'val_visualisation': True,
        'tensorboard': True,
        'model_checkpoint': True,
        'garbage_cleaner': True,
        'whitenning_type': 'simple',
        "vis_frequency": 20
    },

    'data_generator': {
        "single_image": False,
        "batch_size": 16,
        "dim": (256,512),
        'num_regions': 141, 
        "shuffle": True,
        "num_crops": 1,
    },

    'compile_configs': {
        'loss': 'mse'
    },

    'model_configs': {
        'learning_rate': 5e-4,
        'weights': [1, 1],
        'num_regions': 141,
        'run_from_checkpoint': False,
        'checkpoint_dir': "/data/cornucopia/jz522/experiments/2021-05-17/FPN_resnet_bs16_LR5e-4_w11_CONTINUED_2021-05-17_00:25:31/model_checkpoint"
    },

    'fit_model': {
        'epochs': 500,
        'verbose': 1,
    }   
}

with open('configs.json', 'w') as json_file:
    json.dump(configs, json_file, sort_keys=True, indent=4)