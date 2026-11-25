#!/bin/bash
#SBATCH --job-name=f1rand
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# Fully-randomized start group: each episode begins at a group drawn uniformly
# from the curated low-rank pool (Lie rank <= 4). Motivation: the SU(2)-start
# f1fix run stuck at a 0.726 local optimum (never sampled the SM in 4.6M
# episodes) because it committed before exploring the SU(3) parton assignment.
# Randomized starts feed the policy gradient from inside the winning basin.
# Paired with graded terminal bonus (slope, not a +10 exact-only cliff).
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
    --start-mode random \
    --terminal-mode graded --terminal-bonus 3 \
    --run-name f1rand --seed 0
