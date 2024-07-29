#!/bin/bash
# SBATCH --job-name=tiny-1-pool-in-pre

#SBATCH --account=dl65
#SBATCH --partition=m3g

#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00

#SBATCH --mail-user=zizhengpan98@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Command to run a gpu job
# For example:
module load anaconda/2019.03-Python3.7-gcc5
module load gcc/5.4.0
module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
export PROJECT=dl65
export CONDA_ENVS_PATH=/projects/$PROJECT/$USER/conda_envs
export CONDA_PKGS_DIRS=/projects/$PROJECT/$USER/conda_pkgs
source activate /projects/$PROJECT/$USER/conda_envs/defconv
which python


nvidia-smi
nvcc -V
python setup.py build install
