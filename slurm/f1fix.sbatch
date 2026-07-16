#!/bin/bash
#SBATCH --job-name=f1fix
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# Wolfram kernel + GroupMath: load the module (kernel auto-discovered on PATH),
# GroupMath files come from the repo's Mathematica/ by default.
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Shared spectrum cache on lab storage (seeded from the cluster run). All runs
# pool into it and merge on flush, so nothing is ever recomputed.
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl

# metric=f1 + normalization=physical are now the train.py defaults; passed
# explicitly here so this run's provenance is readable from the sbatch alone.
uv run python scripts/train.py \
    --iters 5000 --ckpt-every 50 \
    --metric f1 --normalization physical \
    --run-name f1fix --seed 0
