#!/bin/bash
# Request an hour of runtime:
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --job-name train
#SBATCH -p xhhgnormal01

# module load
module purge
module load compiler/gcc/9.3.0  nvidia/cuda/12.1
cp test.xyz train.xyz
export PATH=/work/share/acitw40es7/softwares/GPUMD-master/src:$PATH
date
nep
date

rm train.xyz 
