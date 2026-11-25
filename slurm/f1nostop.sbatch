#!/bin/bash
#SBATCH --job-name=f1nostop
#SBATCH --partition=shared
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# No-STOP / fixed-horizon variant. STOP is masked out entirely, so the agent can
# never voluntarily commit: an episode ends ONLY at max_steps (128) or the instant
# the model is an exact match to the SM (an absorbing goal). Start group is drawn
# uniformly at random per episode (no curriculum). This is the fixed-horizon,
# absorbing-goal setup common in the RL-for-discovery literature -- it removes the
# early-STOP escape hatch that let f1rand commit to a mediocre 0.64 basin, forcing
# the policy to keep editing the model for the full horizon unless it solves the
# task. Dense delta reward keeps every step informative; the terminal bonus is
# paid only on an exact match (the sole `terminated` event now).
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Shared spectrum cache on lab storage. All runs pool into it and merge on
# flush, so nothing is ever recomputed.
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl

# Jitter kernel launch so jobs co-located on one node don't race the license
# server at the same instant (the engine also retries with a wide STARTUP_TIMEOUT).
sleep $(( RANDOM % 40 ))

uv run python scripts/train.py \
    --iters 5000 --ckpt-every 50 \
    --metric f1 --normalization physical \
    --start-mode random \
    --no-stop --terminal-mode exact --terminal-bonus 5 \
    --run-name f1nostop --seed 0
