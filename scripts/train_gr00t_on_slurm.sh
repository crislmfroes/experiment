#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=results/test_gpu_%j.out
#SBATCH --error=results/test_gpu_%j.err
#SBATCH --gres=gpu
#SBATCH --partition=v100
#SBATCH --mail-user=cris.lima.froes@gmail.com
#SBATCH --mail-type=ALL
git clone https://github.com/crislmfroes/experiment.git
git clone https://github.com/NVIDIA/Isaac-GR00T.git
git clone https://huggingface.co/datasets/crislmfroes/xarm6-sim-pose-cond-v4-pose-rand_v2.0
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --python 3.10 --project Isaac-GR00T
uv run python -c "import gr00t; print('GR00T installed successfully')" --project Isaac-GR00T >> results/out1.txt
#sleep 1m
#nvidia-smi >> results/out1.txt