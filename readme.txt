There are three directories:
DATA = /data/cornucopia/ib255/derivative_datasets/cued_scene_coordinate_regression/data_from_jason/
EXPERIMENTS = /data/cornucopia/jz522/experiments 
PROJECT = /home/mlmi-2020/jz522/localisation_from_image_project


VIRTUAL ENVIRONEMNT:
source /home/mlmi-2020/jz522/localisation_from_image_project/envs/loc/bin/activate

GIT COMMANDS:
git add -A
git commit -m "finalised scene regression model pipeline 2"
git push -u origin master

git diff --stat | tail -n1
git diff --name-only --cached
git reset   # deletes all staged file in the most recent "git add"
git reset HEAD^

# if you want to exclude some directory add this to .gitignore (relative path): <dir_name>/ 

python -m ipykernel install --user --name "localication" --display-name "Python 3 loc"



tensorboard --logdir experiments/logs


TO KILL PROCESSES ON GPU
nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9


git config --global user.email "janekzimoch@gmail.com"
git clone https://gitlab.com/libeigen/eigen.git envs/eigen

mkdir build && cd build && cmake .. -EIGEN_INCLUDE_DIR:STRING="eigen" && make

cmake ../opengv 
  -DEIGEN_INCLUDE_DIR="/home/mlmi-2020/jz522/localisation_from_image_project/envs/eigen"
  -DBUILD_TESTS=ON
  -DCMAKE_INSTALL_PREFIX="/home/mlmi-2020/jz522/localisation_from_image_project/envs"
  -BUILD_PYTHON=ON
make
make install
