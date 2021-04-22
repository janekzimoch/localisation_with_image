from tensorflow.python.client import device_lib
import numpy as np

def get_available_gpus():
    """ call it to get info on names of avialbale GPUs """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def writePlyFile(file_dir, file_name, vertices, colors):
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
    with open(file_dir + file_name, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


def crop_image(image, coords, bearing, mask, orig_dim, start_row=None, start_col=None):
        
    # pick the start cordinates of croped images
    if start_row == None:
        start_row = np.random.randint(0, high=orig_dim[0]-224)
    if start_col == None:
        start_col = np.random.randint(0, high=orig_dim[1])

    # get all pixels that span 224 to the right and down from start pixels
    # if a croped image runs across right image border, then use concatenate to combine pixels of croped image
    if start_col + 224 > 512:
        dif = start_col + 224 - 512
        wraped_image = image[start_row:start_row+224, :dif,:]
        wraped_coords = coords[start_row:start_row+224, :dif,:]
        wraped_bearing = bearing[start_row:start_row+224, :dif,:]
        wraped_mask = mask[start_row:start_row+224, :dif]
        
        croped_image = np.concatenate(
            (image[start_row:start_row+224, start_col:start_col+224,:], wraped_image), axis=1)
        croped_coords = np.concatenate(
            (coords[start_row:start_row+224, start_col:start_col+224,:], wraped_coords), axis=1)
        croped_bearing = np.concatenate(
            (bearing[start_row:start_row+224, start_col:start_col+224,:], wraped_bearing), axis=1)
        croped_mask = np.concatenate(
            (mask[start_row:start_row+224, start_col:start_col+224], wraped_mask), axis=1)

    else:
        croped_image = image[start_row:start_row+224, start_col:start_col+224,:]
        croped_coords = coords[start_row:start_row+224, start_col:start_col+224,:]
        croped_bearing = bearing[start_row:start_row+224, start_col:start_col+224,:]
        croped_mask = mask[start_row:start_row+224, start_col:start_col+224]

    return croped_image, croped_coords, croped_bearing, croped_mask
