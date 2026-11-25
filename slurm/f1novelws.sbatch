#!/bin/bash
#SBATCH --job-name=f1novelws
#SBATCH --partition=sapphire
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# Warm-start counterpart to f1novel2 (Phase 7). CONFIG IS IDENTICAL to f1novel2 --
# the ONLY difference is --init-from: this run seeds its policy from f1nostop's
# final checkpoint (the converged 0.786 plateau policy that discovered the SU3 /
# 3-fundamental-fermion STRUCTURE on its own, unbiased). Weights only; iters,
# optimizer, and best are reset, and the entropy/novelty schedules restart from 0
# to re-inflate exploration from the plateau. This directly tests the Phase-5
# hypothesis -- can gentle novelty cross the ~0.08 charge valley -- starting FROM
# the plateau instead of spending ~1200 iters re-discovering the structure.
#
# Compare/contrast with f1novel2 (fresh, same config): fresh = "can it find and
# then cross"; warm = "given the structure, can it cross". Separate run dirs.
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl
sleep $(( RANDOM % 40 ))

uv run python scripts/train.py \
    --iters 5000 --ckpt-every 50 \
    --init-from runs/f1nostop_20260730_151652/checkpoints/final.pt \
    --metric f1 --normalization physical \
    --start-mode random \
    --no-stop --terminal-mode exact --terminal-bonus 5 \
    --max-steps 48 --rollout 1024 --epochs 2 \
    --ent-coef-final 0.02 \
    --novelty-coef 0.01 --novelty-mode first_visit --novelty-anneal-iters 3000 \
    --run-name f1novelws --seed 0
