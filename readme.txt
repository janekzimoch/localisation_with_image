There are three directories:
DATA = /data/cornucopia/ib255/derivative_datasets/cued_scene_coordinate_regression/data_from_jason/
EXPERIMENTS = /data/cornucopia/jz522/experiments 
PROJECT = /home/mlmi-2020/jz522/localisation_from_image_project


VIRTUAL ENVIRONEMNT:
source /home/mlmi-2020/jz522/localisation_from_image_project/envs/loc/bin/activate

GIT COMMANDS:
git add -A
git commit -m "..."
git push -u origin master

git diff --stat | tail -n1
git diff --name-only --cached
git reset   # deletes all staged file in the most recent "git add"


# if you want to exclude some directory add this to .gitignore (relative path): <dir_name>/ 

python -m ipykernel install --user --name "localication" --display-name "Python 3 loc"



coord_data_dir = "/data/cornucopia/ib255/derivative_datasets/cued_scene_coordinate_regression/data_from_jason/DS_003_JDB-Full/coordinates/"
coord_npz = "0036_rendered.png_config.npz"
# 0036_rendered.png_config.npz
# 0038_rendered.png_config_world.ply
npz_data = np.load(coord_data_dir + coord_npz)


npz_data.files

p_3d_world = npz_data['points_3d_world']
p_3d_camera = npz_data["points_3d_camera"]


points_camera = p_3d_camera.reshape(-1, p_3d_camera.shape[-1])
points_world = p_3d_world.reshape(-1, p_3d_world.shape[-1])