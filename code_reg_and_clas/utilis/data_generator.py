import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    """ Loads, downsamples, crops, and auguments images and labels """
    
    def __init__(self, npz_file_IDs, single_image, batch_size=8, dim=(256,512), num_regions=20, shuffle=True, num_crops=1):

        if single_image:
            self.npz_file_IDs = ["/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512_complete_working_141_classes/0001_rendered.png_config.npz"]
            self.single_crop = True
        else:
            self.npz_file_IDs = npz_file_IDs
            self.single_crop = False

        self.batch_size = batch_size
        self.dim = dim
        self.num_regions = num_regions
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
        images = np.empty((self.batch_size, *self.dim, 3), dtype=np.float32)
        local_scene_coords = np.empty((self.batch_size, *self.dim, 3), dtype=np.float32)
        region_labels = np.empty((self.batch_size, *self.dim, 1), dtype=np.float32)
        masks = np.empty((self.batch_size, *self.dim, 1), dtype=np.int16)

        for i, ID in enumerate(npz_file_ID_temp):
            npz_data = np.load(ID)
            
            images[i] = npz_data['image_colors']
            local_scene_coords[i] = npz_data['local_scene_coords'] #npz_data['points_3d_world'] #
            region_labels[i] = np.expand_dims(npz_data['points_region_class'], axis=-1)
            masks[i] = np.expand_dims(npz_data['mask'], axis=-1).astype(int)

        return images, local_scene_coords, region_labels, masks

    
    def crop(self, data, start_row, start_col, data_depth):
        """ Images are equirectangular (360 projected onto a rectangle).
        Therefore we have to allow for all possibe crops, including these that run across the right image edge. """

        data_croped = np.zeros((self.batch_size, 224, 224, data_depth))

        for ind in range(self.batch_size):

            if start_col[ind] + 224 > 512:
                dif = start_col[ind] + 224 - 512
                wraped_data = data[ind, start_row[ind]:start_row[ind]+224, :dif,:]
                
                data_croped[ind] = np.concatenate(
                    (data[ind, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:], 
                        wraped_data), axis=1)
                               
            else:
                data_croped[ind] = data[ind, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:]

        return data_croped
    
    
    def get_crops(self, images, coords, regions, mask):
        
        # pick the start cordinates of croped images
        if self.single_crop:
            start_row = [0]*self.batch_size
            start_col = [0]*self.batch_size
        else:
            start_row = np.random.randint(0, high=self.dim[0]-224, size=self.batch_size)
            start_col = np.random.randint(0, high=self.dim[1], size=self.batch_size)
        
        image_crops = self.crop(images, start_row, start_col, data_depth=3)
        coords_crops = self.crop(coords, start_row, start_col, data_depth=3)
        region_crops = self.crop(regions, start_row, start_col, data_depth=1)
        mask_crops = self.crop(mask, start_row, start_col, data_depth=1)
        
        return image_crops, coords_crops, region_crops, mask_crops
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.npz_file_IDs) / self.batch_size))
    
     
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.npz_file_IDs[k] for k in indexes]

        # load
        images, local_scene_coords, region_labels, masks = self.load_data(list_IDs_temp)

        # crop
        images, local_scene_coords, region_labels, masks = self.get_crops(images, local_scene_coords, region_labels, masks)

        
        # one-hot encode labels
#         region_labels = (np.arange(self.num_regions) == region_labels[:,:,:,0][...,None]).astype(int)


        # expand dimension of mask
        nested_masks = [masks for _ in range(3)]
        mask_expanded = np.concatenate(nested_masks, axis=-1)

        # apply mask to 3D coords ground truth data
        local_scene_coords = local_scene_coords * mask_expanded
                               
        # coccatenate to images
        labels = np.concatenate((local_scene_coords, region_labels), axis=-1)

        return [images, mask_expanded], labels