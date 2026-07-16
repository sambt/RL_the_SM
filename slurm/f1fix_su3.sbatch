#!/bin/bash
#SBATCH --job-name=f1fix_su3
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# Control run: seed the episode at SU(3) and see whether the agent STAYS there.
# Under the old reward (hungarian + normalization=none) SU(3) was the worst-scoring
# group on the ladder (rank 27/30), so the SU3-seeded run correctly fled to SU8 and
# never returned. Under f1 + physical, SU(3) is no longer repelled (rank 6/30). If
# this run still drifts away from SU(3), the reward fix is incomplete.
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Shared spectrum cache on lab storage (seeded from the cluster run). All runs
# pool into it and merge on flush, so nothing is ever recomputed.
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl

uv run python scripts/train.py \
    --iters 5000 --ckpt-every 50 \
    --metric f1 --normalization physical \
    --run-name f1fix_su3 --start-group SU3 --seed 0
