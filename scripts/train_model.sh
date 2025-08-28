#!/bin/bash -l

#SBATCH --job-name="sd-finetuning"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --account="g96-1840"
#SBATCH --gres=gpu:volta:1
#SBATCH --qos=normal

apptainer exec --nv pytorch-gpu_1.0.sif projects/master-thesis/scripts/run_finetuning.sh "mickuz/dogcatdataset" "sd-finetuned-dogcat"
