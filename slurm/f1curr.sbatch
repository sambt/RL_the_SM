#!/bin/bash
#SBATCH --job-name=f1curr
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# "Easy" curriculum start: anchor on SU(3) early, anneal p(base) 1->0 over the
# first 2000 iters, then fully random over the curated low-rank pool. Gives the
# policy gradient from inside the SU(3) basin (where f1fix_su3 cleanly reached
# 1.0) before asking it to discover the group from scratch. Paired with graded
# terminal bonus (slope, not a +10 exact-only cliff).
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Shared spectrum cache on lab storage. All runs pool into it and merge on
# flush, so nothing is ever recomputed.
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl

# Jitter kernel launch so jobs co-located on one node don't race the license
# server at the same instant (root cause of the first f1rand/f1curr failures;
# the engine also now retries with a wider STARTUP_TIMEOUT as a backstop).
sleep $(( RANDOM % 40 ))

uv run python scripts/train.py \
    --iters 5000 --ckpt-every 50 \
    --metric f1 --normalization physical \
    --start-mode curriculum --curriculum-base SU3 --curriculum-anneal-iters 2000 \
    --terminal-mode graded --terminal-bonus 3 \
    --run-name f1curr --seed 0
