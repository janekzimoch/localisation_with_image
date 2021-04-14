import json
from utilis.experiment_utilis import *




def run_experiment(json_configs_file):
    # get experiment configs
    configs_file = open(json_configs_file)
    configs = json.load(configs_file)
    configs_file.close()

    # setup experiment folder
    print("Setting up Experiment folder ...")
    experiment_name = setup_experiment_folder(**configs['experiment_info'])
    print(experiment_name)
    configs['experiment_info']['experiment_name'] = experiment_name

    # get data IDs
    print("Partitionin gdataset IDs ...")
    data_partition = get_data(**configs['data_partition'])

    # setup generator
    print("Setting up data Generators ...")
    train_generator, validation_generator = get_data_generator(data_partition, configs['data_generator'])

    # setup callbacks
    print("Getting callbacks ...")
    callbacks = get_callbacks(train_generator, validation_generator,
                                **configs['callbacks'], **configs['experiment_info'])

    # setup model
    print("Setting up model ...")
    model = setup_model(configs['setup_model'], **configs['model_configs'])

    # train model
    print("Training ...")
    model.fit(x=train_generator, validation_data=validation_generator, callbacks=callbacks, **configs['fit_model']) 




