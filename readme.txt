There are three directories:
DATA = /data/cornucopia/ib255/derivative_datasets/cued_scene_coordinate_regression/data_from_jason/
EXPERIMENTS = /data/cornucopia/jz522/experiments 
PROJECT = /home/mlmi-2020/jz522/localisation_from_image_project


VIRTUAL ENVIRONEMNT:
source /home/mlmi-2020/jz522/localisation_from_image_project/envs/loc/bin/activate

GIT COMMANDS:
git add -A
git commit -m "added resnet"
git push -u origin master

git diff --stat | tail -n1
git diff --name-only --cached
git reset   # deletes all staged file in the most recent "git add"
git reset HEAD^

# if you want to exclude some directory add this to .gitignore (relative path): <dir_name>/ 

python -m ipykernel install --user --name "localication" --display-name "Python 3 loc"



tensorboard --logdir /data/cornucopia/jz522/experiments/logs


TO KILL PROCESSES ON GPU
nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9


git config --global user.email "janekzimoch@gmail.com"


# INSTALLING openGV
1. git clone openGV
2. git glone eigen git clone https://gitlab.com/libeigen/eigen.git envs/eigen
3. git clone pybind11 to openGV/python
4. run following command:

mkdir build && cd build && cmake ../envs/openGV  \
  -DEIGEN_INCLUDE_DIR="/home/mlmi-2020/jz522/localisation_from_image_project/envs/eigen" \
  -DBUILD_TESTS=ON \
  -DPYBIND11_PYTHON_VERSION=3.7 \
  -DPYTHON_INSTALL_DIR="/home/mlmi-2020/jz522/localisation_from_image_project/envs/loc/lib/python3.7/site-packages" \
  -DCMAKE_INSTALL_PREFIX="/home/mlmi-2020/jz522/localisation_from_image_project/envs"

make
make install





for model computational and memory requirements use model-profiler
from model_profiler import model_profiler

Batch_size = 32
profile = model_profiler(model, Batch_size)

print(profile) 