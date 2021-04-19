from tensorflow import keras
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import k3d
import gc
import os


class RemoveGarbageCallback(keras.callbacks.Callback):
    " Clean garbage variables - hopefully this releases soem memory " 
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


class Visualise_learning(keras.callbacks.Callback):
    """
    Visualises fit of predicted 3D coords for a signle image passed (train_image)
    Saves a .png every "frequency" epochs

    TODO: 
    1) Current visualisation is very 2D. Would be nice to improve its meanigfulness.
    2) See if you can display figures online (integrate with TensorBoard?)    
    """
    def __init__(self, train_image, ground_truth, frequency, exp_name, train_val_setting):
        super(Visualise_learning, self).__init__()
        self.train_image = train_image
        self.ground_truth = ground_truth
        self.frequency = frequency
        self.train_val_setting = train_val_setting
        self.save_dir = '/home/mlmi-2020/jz522/localisation_from_image_project/experiments/' + exp_name + "/train_visualisations/" + train_val_setting
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    

    def plot_simple_3D_point_cloud(self, file_name, ground_truth, pred):
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')

        gt_xyz = ground_truth.reshape(-1, ground_truth.shape[-1])[::20]
        ax.scatter(gt_xyz[:,0], gt_xyz[:,1], gt_xyz[:,2], c='r', marker='o')

        pred_xyz = pred.reshape(-1, pred.shape[-1])[::20]
        ax.scatter(pred_xyz[:,0], pred_xyz[:,1], pred_xyz[:,2], c='b', marker='o')
        
        plt.show()
        fig.savefig(self.save_dir + file_name)


        
    def plot_colored_3D_point_cloud(self, file_name, ground_truth, pred):
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')
        point_subsample=10


        gt_xyz = ground_truth.reshape(-1, ground_truth.shape[-1])[::point_subsample]
        ax.scatter(gt_xyz[:,0], gt_xyz[:,1], gt_xyz[:,2], c='teal', marker='o', label='ground-truth')

        diff = np.abs(ground_truth - pred)
        diff_binary = np.mean(diff, axis=-1)[0]
        diff_binary = diff_binary.reshape(-1)[::20]
        bins = np.array([0.1,1,3])
        binarised = np.digitize(diff_binary, bins, right=False)
        
        pred_xyz = pred.reshape(-1, pred.shape[-1])[::point_subsample]

        colors = ['darkgreen', 'gold', 'tomato', 'darkred']#['darkgreen', 'limegreen', 'gold', 'darkred']
        color_labels = ['awesome [<0.1]', 'good [<1]', 'bad [<3]', 'very bad [+3]']
        for ind, color, label in zip(range(4),colors,color_labels):
            indexes = np.argwhere(binarised==ind)
            ax.scatter(pred_xyz[indexes,0], pred_xyz[indexes,1], pred_xyz[indexes,2], c=color, marker='o', label=label) #RGB
        fig.legend(fontsize=20, loc='upper right', bbox_to_anchor=(0.9, 0.7))

        plt.show()
        fig.savefig(self.save_dir + file_name)


    def plot_pixelwise_coordinate_accuracy(self, file_name, ground_truth, pred):
        """ compute cordinate wise difference  """
        cord_range = np.max(ground_truth) - np.min(ground_truth)

        diff = np.abs(ground_truth - pred)
        diff_binary = np.mean(diff, axis=-1)[0]
        bins = np.array([0.1,1,3])
        binarised = np.digitize(diff_binary, bins, right=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
        
        # LEFT PLOT - shows pixelwise error
        diff_binary[diff_binary > 6] = 6
        im1 = ax1.imshow(diff_binary, cmap='Greys', vmin=0, vmax=6)
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes('right', size='5%', pad=0.1)
        cbar1 = fig.colorbar(im1, cax=cax1)
        ax1.set_title('Pixelwise residual error [gt - pred]')
        cbar1.ax.set_yticklabels(['0','1','2','3','4','5','6+']) 

        # RIGHT PLOT - projection of 3D scatter plot
        im2 = ax2.imshow(binarised, cmap=plt.cm.get_cmap('RdYlGn_r', 4), vmin=-0.5, vmax=3.5)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes('right', size='5%', pad=0.1)
        cbar2 = fig.colorbar(im2, cax=cax2, ticks=[0,1,2,3])
        ax2.set_title('Pixelwise residual error - buckets')
        cbar2.ax.set_yticklabels(['< 0.1', '< 1', '< 3', '3+']) 
        plt.show()
        fig.savefig(self.save_dir + file_name)


        
    def writePlyFile(self, file_name, vertices, colors):
        ply_header = '''ply
                    format ascii 1.0
                    element vertex %(vert_num)d
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                '''
        vertices = vertices.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        vertices = np.hstack([vertices, colors])
        with open(self.save_dir + file_name + '.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(vertices)))
            np.savetxt(f, vertices, '%f %f %f %d %d %d')


            
    def on_epoch_end(self, epoch, logs=None):
        if(epoch%self.frequency == 0):
            train_x = np.expand_dims(self.train_image, axis=0)
            pred_coordinates = self.model.predict(train_x)

            if self.train_val_setting == "val/":
                print()
                print("#############   VALIDATION   #############")
            elif self.train_val_setting == "train/":
                print()
                print("#############     TRAIN     #############")
            
            # plot 3D point convergence
            simple_vis_file = "simple_vis_" + str(epoch) 
            # self.plot_simple_3D_point_cloud(simple_vis_file, self.ground_truth, pred_coordinates)
            self.plot_colored_3D_point_cloud(simple_vis_file, self.ground_truth, pred_coordinates)

            # plot pixel wise accuracy
            pixelwise_acc_file = "pixelwise_acc_" + str(epoch) 
            self.plot_pixelwise_coordinate_accuracy(pixelwise_acc_file, self.ground_truth, pred_coordinates)

            # save .ply file for visualisation
            ply_file = "scene_coordinates_" + str(epoch) 
            self.writePlyFile(ply_file, pred_coordinates, self.train_image)
