from tensorflow.python.client import device_lib
import numpy as np
import os

def get_available_gpus():
    """ call it to get info on names of avialbale GPUs """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def writePlyFile(file_dir, file_name, vertices, colors, sample=0):
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
    
    if sample != 0:
#         print(vertices.shape)
        num_points = int(vertices.shape[0] / sample)
        indices = np.random.choice(vertices.shape[0], size=num_points, replace=False)
        vertices = vertices[indices]
    
    with open(file_dir + file_name, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')




def crop(data, start_row, start_col):
    """ Images are equirectangular (360 projected onto a rectangle).
    Therefore we have to allow for all possibe crops, including these that run across the right image edge. """

    # it assumes start_col and start_row have same length
    num_crops = len(start_row)
    data_depth = data.shape[-1]

    data_croped = np.zeros((num_crops, 224, 224, data_depth))

    for ind in range(num_crops):

        if start_col[ind] + 224 > 512:
            dif = start_col[ind] + 224 - 512
            wraped_data = data[0, start_row[ind]:start_row[ind]+224, :dif,:]
            
            data_croped[ind] = np.concatenate(
                (data[0, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:], 
                    wraped_data), axis=1)
                            
        else:
            data_croped[ind] = data[0, start_row[ind]:start_row[ind]+224, start_col[ind]:start_col[ind]+224,:]

    return data_croped


def stich(data):
    """ takes a (6,224,224,:) dimesnional data and combines 6 crops into a (256,512,:) matrix
    It assumes these are the left-top corners of each crop:
    start_row = [0,0,0,32,32,32]
    start_col = [0,224,448,0,224,448] """
    
    depth = data.shape[-1]
    stiched_data = np.zeros((256,512,depth))

    stiched_data[:224,:224,:]    = data[0,:,:,:]
    stiched_data[:224,224:448,:] = data[1,:,:,:]
    stiched_data[:224,448:,:] = data[2,:,:64,:]
    stiched_data[224:,:224,:] = data[3,192:,:,:]
    stiched_data[224:,224:448,:] = data[4,192:,:,:]
    stiched_data[224:,448:,:] = data[5,192:,:64,:]

    return stiched_data


def get_color_map(regions, colormap=None):
    if colormap == None:
        colromap_dir = "/data/cornucopia/jz522/experiments/model_visualisation/colormap.npy"
        colormap = np.load(colromap_dir)

    reg_flat = np.reshape(regions, (-1)).astype(int)
    colored_regions_flat = colormap[reg_flat]
    colored_regions = np.reshape(colored_regions_flat, (regions.shape[0], regions.shape[1], 3))
    return colored_regions

