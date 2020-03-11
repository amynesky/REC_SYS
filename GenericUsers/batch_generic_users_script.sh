#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --ntasks-per-node 7
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:p100:2
#SBATCH --mail-type=ALL
#SBTACH -e /home/nesky/REC_SYS/GenericUsers
#echo commands to stdout

#echo commands to stdout
set -x


# load modules    #SBATCH -p GPU-shared  #SBATCH -A ac560rp 


#module load hdf5/1.8.16_gnu
module load gcc/5.3.0 boost/1.63.0_py2.7.11 leveldb/1.18 opencv
module load cuda/9.0
module load protobuf/3.2.0
module load cmake
#module load eigen/3.3.4



# run GPU program
./train_generic_users.sh

#The option `--gres-gpu` indicates the number of GPUs you want. 
#The option `--ntasks-per-node` indicates the number of cores you want. 
#It must be greater than or equal to 7 in the GPU-shared partition.



