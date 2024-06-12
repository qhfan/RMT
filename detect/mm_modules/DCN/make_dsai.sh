#!/bin/bash
#SBATCH --account=hhe
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --exclude=node04

#SBATCH --mail-user=zizhengpan98@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Command to run a gpu job
# For example:
module load cuda-11.2.0-gcc-10.2.0-gsjevs3
source activate torch171

nvidia-smi
nvcc -V
python setup.py build install
