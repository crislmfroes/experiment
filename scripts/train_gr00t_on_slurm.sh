#!/bin/bash
#SBATCH --job-name=train_gr00t
#SBATCH --output=results/train_gr00t_%j.out
#SBATCH --error=results/train_gr00t_%j.err
#SBATCH --gres=gpu
#SBATCH --partition=v100
#SBATCH --mail-user=cris.lima.froes@gmail.com
#SBATCH --mail-type=ALL
cd ./Isaac-GR00T
/home/users/crislmfroes/.local/bin/uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /home/users/crislmfroes/xarm6-sim-pose-cond-v4-pose-rand_v2.0 \
    --modality-config-path /home/users/crislmfroes/experiment/source/fbot_arena/fbot_arena/policies/gr00t/xarm6_config.py --embodiment-tag NEW_EMBODIMENT \
    --num-gpus 1 \
    --output-dir /home/users/crislmfroes/results/gr00t_n17_fbot_arena_pose_rand \
    --max-steps 2000 \
    --global-batch-size 1 \
    --dataloader-num-workers 0 --gradient-accumulation-steps 64 --save-total-limit 1
#sleep 1m
#nvidia-smi >> results/out1.txt