from tensorflow import keras
import matplotlib.pyplot as plt
import k3d


class Visualise_learning(keras.callbacks.Callback):
    """
    Visualises fit of predicted 3D coords for a signle image passed (train_image)
    Saves a .png every "frequency" epochs

    TODO: 
    1) Current visualisation is very 2D. Would be nice to improve its meanigfulness.
    2) See if you can display figures online (integrate with TensorBoard?)    
    """
    def __init__(self, train_image, ground_truth, frequency):
        super(Visualise_learning, self).__init__()
        self.train_image = train_image
        self.ground_truth = ground_truth
        self.frequency = frequency
        self.save_dir = '/home/mlmi-2020/jz522/localisation_from_image_project/experiments/exp_1_setup/training_visualisations/'

    def plot_3D_point_cloud(self, pred, ground_truth, name):
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111, projection='3d')

        gt_xyz = ground_truth.reshape(-1, ground_truth.shape[-1])[::20]
        ax.scatter(gt_xyz[:,0], gt_xyz[:,1], gt_xyz[:,2], c='r', marker='o')

        pred_xyz = pred.reshape(-1, pred.shape[-1])[::20]
        ax.scatter(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2], c='b', marker='o')
        
        fig.savefig(self.save_dir + name)

    def on_epoch_end(self, epoch, logs=None):
        if(epoch%self.frequency == 0):
            train_x = np.expand_dims(self.train_image, axis=0)
            pred_coordinates = self.model.predict(train_x)
            file_name = "vis_" + str(epoch) 
            self.plot_3D_point_cloud(pred_coordinates, self.ground_truth, file_name)
