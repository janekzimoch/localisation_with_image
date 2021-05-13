import json

configs = {
    'experiment_info': {
        'experiment_name': "reg__one_crop__old_code",
        'experiment_dir': '/data/cornucopia/jz522/experiments/',
        'multiple_GPU': False,
        'GPUs': ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']
    },
    'data_partition': {
        'dataset_size': 370,
        'data_dir': "/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512_complete_working_141_classes/",
        'val_split': 0.1
    },

    'callbacks': {
        'save_input_images': True,
        'train_visualisation': True,
        'val_visualisation': False,
        'tensorboard': True,
        'model_checkpoint': False,
        'garbage_cleaner': True,
        "vis_frequency": 20
    },

    'data_generator': {
        "single_image": True,
        "batch_size": 1,
        "dim": (256,512),
        "n_channels": 3,
        "shuffle": True,
        "num_crops": 1,
    },

    'compile_configs': {
        'loss': 'mse'
    },

    'model_configs': {
        'learning_rate': 5e-4,
        'run_from_checkpoint': False,
        'checkpoint_dir': "/home/mlmi-2020/jz522/localisation_from_image_project/experiments/2021-04-21/masked_loss_long_2021-04-21_20:59:31/model_checkpoint"
    },

    'fit_model': {
        'epochs': 500,
        'verbose': 1,
    }
}

with open('configs.json', 'w') as json_file:
    json.dump(configs, json_file, sort_keys=True, indent=4)