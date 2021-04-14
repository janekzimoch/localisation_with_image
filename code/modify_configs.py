import json

configs = {
    'experiment_info': {
        'experiment_name': "test",
        'experiment_dir': '/home/mlmi-2020/jz522/localisation_from_image_project/experiments/',
        'multiple_GPU': True
    },
    'data_partition': {
        'dataset_size': 370,
        'data_dir': "/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512/",
        'val_split': 0.05
    },

    'callbacks': {
        'train_visualisation': True,
        'val_visualisation': True,
        'tensorboard': True,
        'model_checkpoint': True,
        'garbage_cleaner': True,

    },

    'data_generator': {
        "batch_size": 4,
        "dim": (256,512),
        "n_channels": 3,
        "shuffle": True,
        "num_crops": 8,
    },

    'setup_model': {
        'loss': 'mse', 
    },

    'model_configs': {
        'learning_rate': 0.0001
    },

    'fit_model': {
        'epochs': 2,
        'verbose': 1,
    }
}

with open('configs.json', 'w') as json_file:
    json.dump(configs, json_file, sort_keys=True, indent=4)