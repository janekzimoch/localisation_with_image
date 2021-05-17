import tensorflow as tf
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


class Save_sample_input(keras.callbacks.Callback):
    def __init__(self, input_data, labels, exp_dir, exp_name):
        super(Save_sample_input, self).__init__()
        self.images, self.mask = input_data
        self.ground_truth = labels
        self.save_dir = exp_dir + exp_name + "/sample_input/"
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_train_begin(self, logs=None):
        i = 0
        for image, mask, gt in zip(self.images[:10], self.mask[:10], self.ground_truth[:10]):
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,3))

            ax1.imshow(image)
            ax1.set_title("input image")

            ax2.imshow(mask[:,:,0])
            ax2.set_title("mask")

            local_coords_norm = (gt[:,:,:3] - gt[:,:,:3].min()) / (gt[:,:,:3].max() - gt[:,:,:3].min())
            local_coords_norm = local_coords_norm * mask[0]
            ax3.imshow(local_coords_norm)
            ax3.set_title("local scene coords (norm)")

            ax4.imshow(gt[:,:,3:])
            ax4.set_title("region labels")
            
            file_name = f"input_visualisation_{i}"
            fig.savefig(self.save_dir + file_name, facecolor='w')
            plt.close()
            i += 1

        

class Visualise_learning_reg_and_class(keras.callbacks.Callback):
    def __init__(self, image, mask, gt_coords, gt_classes, frequency, exp_dir, exp_name, train_val_setting):
        super(Visualise_learning_reg_and_class, self).__init__()

        self.image = image
        self.mask = mask
        self.gt_coords = gt_coords
        self.gt_classes = gt_classes

        self.frequency = frequency
        self.train_val_setting = train_val_setting
        self.save_dir = exp_dir + exp_name + "/train_visualisations/" + train_val_setting
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def visualise_distributions(self, file_name, label_pred):

        # visualise distribution of TRUE and PRED
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,3))

        ax1.hist(np.reshape(label_pred[0,:,:,:3], (-1)), label="pred", alpha=0.3)
        ax1.hist(np.reshape(self.gt_coords, (-1)), label="true", alpha=0.3)
        ax1.legend()

        ax2.hist(np.reshape(np.argmax(label_pred[0,:,:,3:], axis=-1), (-1)), label="pred", alpha=0.3, bins=20)
        ax2.hist(np.reshape(self.gt_classes, (-1)), label="true", alpha=0.3, bins=20)
        ax2.legend()
        
        plt.show()
        fig.savefig(self.save_dir + file_name, facecolor='w')        


    def visualise_output(self, file_name, label_pred):
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(20,3))

        ax1.imshow(self.image)
        ax1.set_title("input image")

        ax2.imshow(self.mask[:,:,0])
        ax2.set_title("mask")
        
        pred_norm_local_cords = (label_pred[0,:,:,:3] - label_pred[0,:,:,:3].min()) / (label_pred[0,:,:,:3].max() - label_pred[0,:,:,:3].min())
        ax3.imshow(pred_norm_local_cords)
        ax3.set_title("local scene coords (norm)")

        ax4.imshow(np.argmax(label_pred[0,:,:,3:], axis=-1))
        ax4.set_title("region labels")

        gt_norm_local_cords = (self.gt_coords - self.gt_coords.min()) / (self.gt_coords.max() - self.gt_coords.min())
        ax5.imshow(gt_norm_local_cords)
        ax5.set_title("GT local scene coords (norm)")

        ax6.imshow(self.gt_classes)
        ax6.set_title("GT region labels")
        
        plt.show()
        fig.savefig(self.save_dir + file_name, facecolor='w')


    def on_epoch_begin(self, epoch, logs=None):
        if(epoch%self.frequency == 0):
            # GET PREDICTION
            image = np.expand_dims(self.image, axis=0)
            mask = np.expand_dims(self.mask, axis=0)

            output = self.model.predict([image, mask])

            # VISUALISE
            if self.train_val_setting == "val/":
                print("\n ###   VALIDATION   ###")
            elif self.train_val_setting == "train/":
                print("\n ###     TRAIN      ###")
            
            output_vis_file_name = "output_vis_" + str(epoch) 
            self.visualise_output(output_vis_file_name, output)

            distribution_file_name = "distribution_" + str(epoch) 
            self.visualise_distributions(distribution_file_name, output)



class pixelwise_MSE(keras.callbacks.Callback):
    def __init__(self, datapoint_name, metric_frequency, visualisation_frequency, exp_dir, exp_name, start_row=[0], start_col=[0]):
        super(pixelwise_MSE, self).__init__()
        
        self.writer = tf.summary.create_file_writer(exp_dir + 'logs/' + exp_name + "_MSE")
        self.step_number = 0
        data = self.load_data(datapoint_name, start_row, start_col)
        
        self.image = data['image']
        self.oracle_global_coords = data['points_3d_world']  # i will use this to check whether i'm converting the data correctly
        self.mask = data['mask']
        self.W_inv = data['W_inv']
        self.M = data['M']
        self.std = data['std']
        
        self.visualisation_frequency = visualisation_frequency
        self.metric_frequency = metric_frequency
        
        
      
    def crop(self, data, start_row, start_col, data_depth):
        num_crops = len(start_row)
        data_croped = np.zeros((num_crops, 224, 224, data_depth))

        for ind in range(num_crops):
            if start_col[ind] + 224 > 512:
                dif = start_col[ind] + 224 - 512
                wraped_data = data[ind, start_row[ind]:start_row[ind]+224, :dif,:]
                
                data_croped[ind] = np.concatenate(
                    (data[ind, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:], 
                        wraped_data), axis=1)  
            else:
                data_croped[ind] = data[ind, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:]

        return data_croped
    
    
    def load_data(self, data_filename, start_row, start_col):
        data = np.load(data_filename)

        image = self.crop(  np.expand_dims(data['image_colors'], axis=0), start_row, start_col, 3)
        points_3d_world = self.crop(  np.expand_dims(data['points_3d_world'], axis=0), start_row, start_col, 3)
        mask = self.crop(  np.expand_dims(data['mask'], axis=[0,-1]), start_row, start_col, 1)
        
        W_inv = data['W_inv']
        M = data['M']
        std = data['std']
        
        
        return {'image': image,
                'points_3d_world': points_3d_world, 
                'mask': mask,
                'W_inv': W_inv,
                'M': M, 
                'std': std}
    
        
    def on_epoch_begin(self, epoch, logs=None):
        if(epoch%self.metric_frequency == 0):

            output = self.model.predict([self.image, self.mask])
            global_coords = np.zeros((224*224,3))
            
            # 1. convert regions to their mean centers
            pred_local_coords = np.reshape(output[:,:,:,:3], (-1,3))
            labels = np.argmax(output[:,:,:,3:], axis=-1)
            pred_regions = np.reshape(labels, (-1)).astype(int)
                       
          
            # 2. unwhitten local coordinates
            for region in np.unique(pred_regions):
                region_coords = pred_local_coords[pred_regions == region]
                unwhite_loc_coords = np.dot(region_coords * self.std[region] , self.W_inv[region]) + self.M[region]
                
                global_coords[pred_regions == region] = unwhite_loc_coords
            
           
            # 4. compute MSE
            global_coords = np.reshape(global_coords, (224,224,3))
            global_coords = global_coords * self.mask[0]
            oracle = self.oracle_global_coords[0] * self.mask[0]
            errors_squared = np.square(oracle - global_coords)
            euc_dist = np.mean(np.square(oracle - global_coords), axis=-1)
            mean_euc_dist = np.mean(euc_dist)
            print('MSE: ', mean_euc_dist)
            
            # 5. compute MSE 90th percentile
            percentile_90th = np.percentile(errors_squared, 90)
            euc_dist_perc_90th = np.mean(errors_squared[errors_squared < percentile_90th])
            mean_euc_dist_90th = np.mean(euc_dist_perc_90th)
            print('MSE 90th percentile: ', mean_euc_dist_90th)
            
            

            # 6. save to tensorboard
            with self.writer.as_default():
                tf.summary.scalar('MSE', mean_euc_dist, step=self.step_number, description=None)
                tf.summary.scalar('MSE_90th_perc', mean_euc_dist_90th, step=self.step_number, description=None)
            self.step_number += 1

            
            # 6. plot for visualisation
            if(epoch%self.visualisation_frequency == 0):
                
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,3))
                region_centers = self.M[pred_regions]
                max_ = max(self.oracle_global_coords[0].max(), global_coords.max())
                min_ = min(self.oracle_global_coords[0].min(), global_coords.min())
                
                oracle_coords_norm = (self.oracle_global_coords[0] - min_) / (max_ - min_)
                ax1.imshow(oracle_coords_norm)
                ax1.set_title("Ground truth - global scene coords")
                
                pred_coords_norm = (global_coords - min_) / (max_ - min_)
                ax2.imshow(pred_coords_norm)
                ax2.set_title("Pred - global scene coords")
                
                region_centers = np.reshape(region_centers, (224,224,3))
                ax3.imshow(region_centers)
                
                im4 = ax4.imshow(euc_dist)
                ax4.set_title("Euclidean distance")
                
                divider = make_axes_locatable(ax4)
                cax = divider.append_axes('right', size='5%', pad=0.1)
                cbar = fig.colorbar(im4, cax=cax)

                plt.show()



class pixelwise_MSE_agregate(keras.callbacks.Callback):
    def __init__(self, generator, datapoint_name, metric_frequency, visualisation_frequency, exp_dir, exp_name, train, start_row=[0], start_col=[0]):
        super(pixelwise_MSE_agregate, self).__init__()
        
        # store data to print at the end
        self.MSE = []
        self.MSE_90th = []

        # tensorboard writer
        self.writer = tf.summary.create_file_writer(exp_dir + 'logs/' + exp_name + "_MSE")
        self.step_number = 0
        
        # get unwhitenning params
        data = self.load_data(datapoint_name, start_row, start_col)
        self.W_inv = data['W_inv']
        self.M = data['M']
        self.std = data['std']
        
        self.generator = generator
        self.visualisation_frequency = visualisation_frequency
        self.metric_frequency = metric_frequency
        if train:
            self.type = "_train"
        else:
            self.type = '_val'

    def get_metric(self):
        return self.MSE, self.MSE_90th
        
    def load_data(self, data_filename, start_row, start_col):
        data = np.load(data_filename)

        W_inv = data['W_inv']
        M = data['M']
        std = data['std']
        
        return {'W_inv': W_inv, 'M': M, 'std': std}
    
        
    def on_epoch_begin(self, epoch, logs=None):
        if(epoch%self.metric_frequency == 0):


            # get data
            [images, mask], labels = self.generator.__getitem__(0)
            self.generator.on_epoch_end() # shuffle order

            batch_size = len(labels)

            output = self.model.predict([images, mask])
            global_coords = np.zeros((batch_size*224*224,3))
            oracle_global_coords = np.zeros((batch_size*224*224,3))
            
                       
          
            # 2. unwhitten local coordinates - pred
            pred_local_coords = np.reshape(output[:,:,:,:3], (-1,3))
            pred_regions = np.argmax(output[:,:,:,3:], axis=-1)
            pred_regions = np.reshape(pred_regions, (-1)).astype(int)

            for region in np.unique(pred_regions):
                region_coords = pred_local_coords[pred_regions == region]
                unwhite_loc_coords = np.dot(region_coords * self.std[region] , self.W_inv[region]) + self.M[region]
                
                global_coords[pred_regions == region] = unwhite_loc_coords


            # 2. unwhitten local coordinates - GT oracle
            oracle_local_coords = np.reshape(labels[:,:,:,:3], (-1,3))
            oracle_regions = np.reshape(labels[:,:,:,3:], (-1)).astype(int)

            for region in np.unique(oracle_regions):
                region_coords = oracle_local_coords[oracle_regions == region]
                unwhite_loc_coords = np.dot(region_coords * self.std[region] , self.W_inv[region]) + self.M[region]
                
                oracle_global_coords[oracle_regions == region] = unwhite_loc_coords
            
           
            # 4. compute MSE
            global_coords = np.reshape(global_coords, (batch_size,224,224,3))
            oracle_global_coords = np.reshape(oracle_global_coords, (batch_size,224,224,3))
            global_coords = global_coords * mask
            oracle = oracle_global_coords * mask
            errors_squared = np.square(oracle - global_coords)
            euc_dist = np.mean(np.square(oracle - global_coords), axis=-1)
            mean_euc_dist = np.mean(euc_dist)
            print(f'MSE{self.type}: ', mean_euc_dist)
            self.MSE.append(mean_euc_dist)
            
            # 5. compute MSE 90th percentile
            percentile_90th = np.percentile(errors_squared, 90)
            euc_dist_perc_90th = np.mean(errors_squared[errors_squared < percentile_90th])
            mean_euc_dist_90th = np.mean(euc_dist_perc_90th)
            print(f'MSE{self.type} 90th percentile: ', mean_euc_dist_90th)
            self.MSE_90th.append(mean_euc_dist_90th)
            
            

            # 6. save to tensorboard
            with self.writer.as_default():
                tf.summary.scalar('MSE' + self.type, mean_euc_dist, step=self.step_number, description=None)
                tf.summary.scalar('MSE_90th_perc' + self.type, mean_euc_dist_90th, step=self.step_number, description=None)
            self.step_number += 1

            
            # 6. plot for visualisation
            if(epoch%self.visualisation_frequency == 0):
                
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,3))
                region_centers = self.M[pred_regions]
                max_ = max(oracle_global_coords[0].max(), global_coords[0].max())
                min_ = min(oracle_global_coords[0].min(), global_coords[0].min())
                
                oracle_coords_norm = (oracle_global_coords[0] - min_) / (max_ - min_)
                ax1.imshow(oracle_coords_norm)
                ax1.set_title("Ground truth - global scene coords")
                
                pred_coords_norm = (global_coords[0] - min_) / (max_ - min_)
                ax2.imshow(pred_coords_norm)
                ax2.set_title("Pred - global scene coords")
                
                region_centers = np.reshape(region_centers, (batch_size, 224,224,3))
                ax3.imshow(region_centers[0])
                ax3.set_title("Region Centers")
                
                im4 = ax4.imshow(euc_dist[0])
                ax4.set_title("Euclidean distance")
                
                divider = make_axes_locatable(ax4)
                cax = divider.append_axes('right', size='5%', pad=0.1)
                cbar = fig.colorbar(im4, cax=cax)

                plt.show()
            



class Visualise_learning_regression(keras.callbacks.Callback):
    """
    Visualises fit of predicted 3D coords for a signle image passed (train_image)
    Saves a .png every "frequency" epochs
   
    """
    def __init__(self, image, mask, gt_coords, gt_classes, frequency, exp_dir, exp_name, train_val_setting):
        super(Visualise_learning_regression, self).__init__()
        self.image = image
        self.mask = mask
        self.gt_coords = gt_coords
        self.gt_classes = gt_classes
        self.frequency = frequency
        self.train_val_setting = train_val_setting
        self.save_dir = exp_dir + exp_name + "/train_visualisations/" + train_val_setting
        
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


        
    def plot_colored_3D_point_cloud(self, file_name, ground_truth, pred, mask):
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')
        point_subsample=10

        gt_xyz = ground_truth[mask[0]]
        gt_xyz = gt_xyz.reshape(-1, ground_truth.shape[-1])
        gt_xyz = gt_xyz[::point_subsample]
        ax.scatter(gt_xyz[:,0], gt_xyz[:,1], gt_xyz[:,2], c='aqua', alpha=0.5, marker='o', label='ground-truth')

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
        cbar1.ax.set_yticklabels(['Masked / 0','1','2','3','4','5','6+']) 

        # RIGHT PLOT - projection of 3D scatter plot
        im2 = ax2.imshow(binarised, cmap=plt.cm.get_cmap('RdYlGn_r', 4), vmin=-0.5, vmax=3.5)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes('right', size='5%', pad=0.1)
        cbar2 = fig.colorbar(im2, cax=cax2, ticks=[0,1,2,3])
        ax2.set_title('Pixelwise residual error - buckets')
        cbar2.ax.set_yticklabels(['< 0.1', '< 1', '< 3', '3+']) 
        plt.show()
        fig.savefig(self.save_dir + file_name)


        
    def writePlyFile(self, file_name, vertices, colors, mask):

        # remove 3D points which should be masked
        vertices = vertices[mask]
        colors = colors[mask[0]]

        # write
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


            
    def on_epoch_begin(self, epoch, logs=None):
        if(epoch%self.frequency == 0):
            image = np.expand_dims(self.image, axis=0)
            mask = np.expand_dims(self.mask, axis=0)

            output = self.model.predict([image, mask])
            pred_local_coordinates = output[0,:,:,:3]

            if self.train_val_setting == "val/":
                print()
                print("#############   VALIDATION   #############")
            elif self.train_val_setting == "train/":
                print()
                print("#############     TRAIN     #############")
            
            # plot 3D point convergence
            simple_vis_file = "simple_vis_" + str(epoch) 
            # self.plot_simple_3D_point_cloud(simple_vis_file, self.ground_truth, pred_coordinates)
            self.plot_colored_3D_point_cloud(simple_vis_file, self.gt_coords, pred_local_coordinates, mask.astype(bool))

            # plot pixel wise accuracy
            pixelwise_acc_file = "pixelwise_acc_" + str(epoch) 
            self.plot_pixelwise_coordinate_accuracy(pixelwise_acc_file, self.gt_coords, pred_local_coordinates)

            # save .ply file for visualisation
            ply_file = "scene_coordinates_" + str(epoch) 
            self.writePlyFile(ply_file, pred_local_coordinates, self.image, mask.astype(bool))
