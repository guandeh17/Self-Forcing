#!/bin/bash
#SBATCH --job-name=la-tests
#SBATCH --account=bhnr-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/la-tests-%j.out
set -uo pipefail
cd /projects/bhnr/tdao3/workspace/self-forcing
export TORCHINDUCTOR_CACHE_DIR=/scratch/bhnr/tdao3/.inductor_cache
export TRITON_CACHE_DIR=/scratch/bhnr/tdao3/.triton_cache
nvidia-smi -L
.venv/bin/python tests/test_lookahead.py
