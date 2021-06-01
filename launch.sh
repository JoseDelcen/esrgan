#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=28
#SBATCH -c 1
#SBATCH --time=0-08:00:00
#SBATCH -p batch
print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"
export MODULEPATH=/opt/apps/resif/iris/2019b/gpu/modules/all
module load lang/Python/3.7.4-GCCcore-8.3.0
source envs/esrgan/bin/activate
python train.py
