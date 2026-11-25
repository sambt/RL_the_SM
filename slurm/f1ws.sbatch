#!/bin/bash
#SBATCH --job-name=f1ws
#SBATCH --partition=sapphire
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# Phase 8: the warm start done properly. f1novelws failed because --init-from
# restarted ent_coef at 0.03 (5x what the seed converged under) and the novelty
# bonus added a second diversity force; the seed diffused to near-uniform in
# ~250 iters and lost the SU3/3-fund structure. This run:
#   seed  = f1nostop ckpt_004500 (best 50-iter mean return 0.71, entropy 0.79;
#           final.pt came after an entropy collapse to 0.19 and a return dip)
#   ent   = 0.01 FIXED (--ent-anneal-iters 0), ~the coefficient the seed was
#           trained under at iter 4000-4500 (0.0075-0.010): gentle perturbation
#   novelty OFF (default 0)
#   epochs 2, max_steps 48, rollout 1024 as in the Phase 7 bundle.
# Question: does a gently perturbed plateau policy cross the ~0.08 charge valley
# on its own? Watch: iter-0 best ~0.7, entropy staying ~0.8-1.1 (NOT climbing to
# 2+), best_score passing 0.786, any exact hits.
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl
sleep $(( RANDOM % 40 ))

uv run python scripts/train.py \
    --iters 5000 --ckpt-every 50 \
    --init-from runs/f1nostop_20260730_151652/checkpoints/ckpt_004500.pt \
    --metric f1 --normalization physical \
    --start-mode random \
    --no-stop --terminal-mode exact --terminal-bonus 5 \
    --max-steps 48 --rollout 1024 --epochs 2 \
    --ent-coef 0.01 --ent-coef-final 0.01 --ent-anneal-iters 0 \
    --run-name f1ws --seed 0
