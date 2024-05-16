#!/bin/bash
## Request GPU partitions
#SBATCH -p gpu
## Request the number of GPUs to be used (if more than 1 GPU is required, change 1 into Ngpu, where Ngpu=2,3,4)
#SBATCH --gres=gpu:4
## Request the number of nodes
#SBATCH -N 1
## Request the number of CPU cores (There are 24 cores on the GPU node, so 6 cores for 1 GPU)
#SBATCH -n 6
#SBATCH -o SAVSNET_test/test_result.txt
source savsnet_hpc/bin/activate
python3 SAVSNET_test.py
