#!/bin/bash

#SBATCH --job-name=1
#SBATCH --partition=gpuB
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
seu_share/home/huangjie/220235144/anaconda3/envs/Reefknot/bin/python extract_layer_probs_hallu.py   --model-path /path/to/llava-v1.5-7b   --model-base None   --yesno-file results/llava_answer.jsonl   --output-dir hallu_cases_layer_probs   --conv-mode llava_v1   --num-samples 100