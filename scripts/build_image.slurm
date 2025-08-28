#!/bin/bash -l
#SBATCH --job-name="container_image_build"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=g96-1840
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --qos=normal

export TMPDIR=/home/mk437922/tmp/
apptainer pull docker://mickuz/pytorch-gpu:$1
