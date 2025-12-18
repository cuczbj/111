#!/bin/bash

#SBATCH --job-name=1
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -o %j.out
#SBATCH -e %j.err
module load anaconda3
source activate base
cd /seu_share/home/huangjie/220235144/spatial-relation-benchmark-main
source /seu_share/home/huangjie/220235144/anaconda3/etc/profile.d/conda.sh
conda activate srp
python scripts/spatialsense_regionvit.py