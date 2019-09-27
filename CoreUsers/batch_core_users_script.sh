#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU
#SBATCH --ntasks-per-node 7
#SBATCH --gres=gpu:p100:2
#SBATCH -t 48:00:00
#SBATCH --mail-type=ALL
#echo commands to stdout

#echo commands to stdout
set -x


# load modules    #SBATCH -p GPU-shared #SBATCH -A ac560rp 


module load gcc/5.3.0 boost/1.63.0_py2.7.11 leveldb/1.18 opencv
module load cuda/8.0 protobuf/3.2.0
module load cmake

# run GPU program
./train_core_users.sh

#The option `--gres-gpu` indicates the number of GPUs you want. 
#The option `--ntasks-per-node` indicates the number of cores you want. 
#It must be greater than or equal to 7 in the GPU-shared partition.



