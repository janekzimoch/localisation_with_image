from tempfile import TemporaryFile
from tqdm import tqdm
import numpy as np
import os
import cv2


if __name__ == '__main__':
    coord_data_dir = "/data/cornucopia/ib255/derivative_datasets/cued_scene_coordinate_regression/data_from_jason/DS_003_JDB-Full/coordinates/"
    dir_to_save = "/data/cornucopia/jz522/localisation_project/DS_003_JDB-Full/coordinates_256_512_full/"

    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)


    for ind in tqdm(range(350,370)):
        coord_npz =  f"{ind:04}_rendered.png_config.npz"
        npz_data = np.load(coord_data_dir + coord_npz)

        colors = np.zeros((256, 512, 3))
        scene_coordinates = np.zeros((256, 512, 3))
        # camera_coordinates = np.zeros((256, 512, 3))
        camera_bearings = np.zeros((256, 512, 3))

        colors = cv2.resize(npz_data['image_colors'], (512, 256), interpolation = cv2.INTER_CUBIC)
        scene_coordinates = cv2.resize(np.array(npz_data['points_3d_world'], dtype=np.float32), (512, 256), interpolation = cv2.INTER_NEAREST)
        # camera_coordinates = cv2.resize(np.array(npz_data['points_3d_camera'], dtype=np.float32), (512, 256), interpolation = cv2.INTER_NEAREST)  # , points_3d_camera=camera_coordinates
        camera_bearings = cv2.resize(np.array(npz_data['points_3d_sphere'], dtype=np.float32), (512, 256), interpolation = cv2.INTER_NEAREST)

        np.savez(dir_to_save + coord_npz, image_colors=colors, points_3d_world=scene_coordinates, points_3d_sphere=camera_bearings, R_blender=npz_data['R_blender'], T_blender=npz_data['T_blender'])
