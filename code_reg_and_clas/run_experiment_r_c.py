from utilis.experiment_utilis import *
import json


def run_experiment(json_configs_file):
    
    # get experiment configs
    configs = get_configs(json_configs_file)
    experiment_info = configs['experiment_info']

    # decide on multi GPU or single GPU
    if experiment_info['multiple_GPU'] == True:
        strategy = tf.distribute.MirroredStrategy(experiment_info['GPUs'])
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            history, metrics = run_setup(configs)
    else:
        history, metrics = run_setup(configs)
        # run_setup(configs)

    return history, metrics

def run_setup(configs):
    
    # setup experiment folder
    print("Setting up Experiment folder ...")
    experiment_name, experiment_full_name = setup_experiment_folder(**configs['experiment_info'])
    print(experiment_name)
    configs['experiment_info']['experiment_name'] = experiment_name

    # get data IDs
    print("Partitioning dataset IDs ...")
    data_partition = get_data(experiment_full_name, **configs['data_partition'], **configs['model_configs'])

    # setup generator
    print("Setting up data Generators ...")
    train_generator, validation_generator = get_data_generator(data_partition, configs['data_generator'])

    # setup callbacks
    print("Getting callbacks ...")
    callbacks, MSE_train, MSE_val = get_callbacks(train_generator, validation_generator,
                                **configs['callbacks'], **configs['experiment_info'])
    # , MSE_train, MSE_val 
    
    # setup model
    print("Setting up model ...")
    model = setup_model(configs['compile_configs'], **configs['model_configs'])

    # train model
    print("Training ...")
    history = model.fit(x=train_generator, validation_data=validation_generator, callbacks=callbacks, **configs['fit_model']) 

    # save history
    save_history(experiment_full_name, history.history)

    # save model
    model_dir = experiment_full_name + "/saved_model"
    os.makedirs(model_dir)
    model.save(model_dir + '/my_model')

    # extract MSE values from the Callback object
    train_MSE, train_MSE_90th = MSE_train.get_metric()
    val_MSE, val_MSE_90th = MSE_val.get_metric()
    MSE_metrics = {'train_MSE': train_MSE, 'train_MSE_90th': train_MSE_90th, 'val_MSE': val_MSE, 'val_MSE_90th': val_MSE_90th}

    # save those metrics to a json file
    with open(experiment_full_name + '/MSE_metric.json', 'w') as json_file:
        json.dump(MSE_metrics, json_file, sort_keys=True, indent=4)

    return history, [train_MSE, train_MSE_90th, val_MSE, val_MSE_90th]


