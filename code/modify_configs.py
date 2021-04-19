import json

configs = {
    'experiment_info': {
        'experiment_name': "full_run_world_coord",
        'experiment_dir': '/home/mlmi-2020/jz522/localisation_from_image_project/experiments/',
        'multiple_GPU': True,
        'GPUs': ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']
    },
    'data_partition': {
        'dataset_size': 370,
        'data_dir': "/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512_full/",
        'val_split': 0.1
    },

    'callbacks': {
        'train_visualisation': True,
        'val_visualisation': True,
        'tensorboard': True,
        'model_checkpoint': True,
        'garbage_cleaner': True,
        "vis_frequency": 20
    },

    'data_generator': {
        "batch_size": 20,
        "dim": (256,512),
        "n_channels": 3,
        "shuffle": True,
        "num_crops": 8,
    },

    'compile_configs': {
        'loss': 'mse'
    },

    'model_configs': {
        'learning_rate': 0.0000005,
        'run_from_checkpoint': True,
        'checkpoint_dir': "/home/mlmi-2020/jz522/localisation_from_image_project/experiments/2021-04-18/full_run_world_coord_2021-04-18_11:28:32/model_checkpoint"
    },

    'fit_model': {
        'epochs': 200,
        'verbose': 1,
    }
}

with open('configs.json', 'w') as json_file:
    json.dump(configs, json_file, sort_keys=True, indent=4)