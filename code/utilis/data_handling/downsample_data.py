from tempfile import TemporaryFile
import numpy as np
import os
import cv2


def get_file_name(ind):
    if ind < 10:
        ind = "00" + str(ind)
    elif ind < 100:
        ind = "0" + str(ind)

    coord_npz = "0{}_rendered.png_config.npz".format(str(ind))
    return coord_npz

if __name__ == '__main__':
    coord_data_dir = "/data/cornucopia/ib255/derivative_datasets/cued_scene_coordinate_regression/data_from_jason/DS_003_JDB-Full/coordinates/"
    dir_to_save = "/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512/"

    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)


    for ind in range(1,371):
        coord_npz = get_file_name(ind)
        npz_data = np.load(coord_data_dir + coord_npz)

        image_low = np.zeros((256, 512, 3))
        label_low = np.zeros((256, 512, 3))

        image_low = cv2.resize(npz_data['image_colors'], (512, 256), interpolation = cv2.INTER_CUBIC)
        label_low = cv2.resize(np.array(npz_data['points_3d_camera'], dtype=np.float32), (512, 256), interpolation = cv2.INTER_NEAREST)

        file_name = f"{ind:03}_rendered.png_config.npz"
        np.savez(dir_to_save + file_name, image_colors=image_low, points_3d_camera=label_low)
