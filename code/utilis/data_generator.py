import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    """ Loads, downsamples, crops, and auguments images and labels """
    
    def __init__(self, npz_file_IDs, batch_size=8, dim=(256,512), n_channels=3, shuffle=True, num_crops=4):

        self.npz_file_IDs = npz_file_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.num_crops = num_crops

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.npz_file_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
            
    def load_data(self, npz_file_ID_temp):
        """ Load .npz files with images and labels from self.room_dir """
        images = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.int16)
        labels = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        for i, ID in enumerate(npz_file_ID_temp):
            npz_data = np.load(ID)
            
            images[i] = npz_data['image_colors'].astype(int)
            labels[i] = npz_data['points_3d_world']

        return images, labels
            
            
    def load_and_downsample(self, npz_file_ID_temp):
        """ We choose to downsample to 256x512 and then crop 224x224 images. Downsampling to
        256x512 has proven to work well in practice, although we ourselves haven't tested that statement."""
        images = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.int16)
        labels = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        

        for i, ID in enumerate(npz_file_ID_temp):
            npz_data = np.load(ID)

            images[i,:,:,:] = cv2.resize(npz_data['image_colors'], self.dim[::-1], interpolation = cv2.INTER_CUBIC)
            labels[i,:,:,:] = cv2.resize(np.array(npz_data['points_3d_world'], dtype=np.float32), self.dim[::-1], interpolation = cv2.INTER_NEAREST)

        return images, labels

    
    def get_image_crops(self, images, labels):
        """ Images are equirectangular (360 projected onto a rectangle).
        Therefore we have to allow for all possibe crops, including these that run across the right image edge. """

        image_crops = np.zeros((self.batch_size, 224, 224, self.n_channels), dtype=np.int16)
        label_crops = np.zeros((self.batch_size, 224, 224, self.n_channels), dtype=np.float32)

        # pick the start cordinates of croped images
        start_row = np.random.randint(0, high=self.dim[0]-224, size=self.batch_size)
        start_col = np.random.randint(0, high=self.dim[1], size=self.batch_size)

        # get all pixels that span 224 to the right and down from start pixels
        for ind in range(self.batch_size):

            if start_col[ind] + 224 > 512:
                dif = start_col[ind] + 224 - 512
                wraped_image = images[ind, start_row[ind]:start_row[ind]+224, :dif,:]
                wraped_label = labels[ind, start_row[ind]:start_row[ind]+224, :dif,:]
                
                image_crops[ind] = np.concatenate(
                    (images[ind, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:], 
                        wraped_image), axis=1)
                label_crops[ind] = np.concatenate(
                    (labels[ind, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:], 
                        wraped_label), axis=1)

            else:
                image_crops[ind] = images[ind, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:]
                label_crops[ind] = labels[ind, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:]


        return image_crops, label_crops
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.npz_file_IDs) / self.batch_size))
    
    def get_mask(self, labels):
        sumed_coords = np.sum(labels, axis=-1)
        mask = np.where(sumed_coords == 0, 0, 1)
        dim = mask.shape
        mask = np.reshape(mask, (dim[0], dim[1], dim[2], 1) )

        return mask
    
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.npz_file_IDs[k] for k in indexes]

        # load images (from a downsampled dataset)
        images, labels= self.load_data(list_IDs_temp)
        
        # Get images and labels (downsampled)
        # images, labels = self.load_and_downsample(list_IDs_temp)

        # crop
        images, labels = self.get_image_crops(images, labels)
    
        # get mask
        mask = self.get_mask(labels)

        # coccatenate to images
        labels = np.concatenate((labels, mask), axis=-1)

        return images, labels