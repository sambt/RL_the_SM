#!/bin/bash
#SBATCH --job-name=f1novel_rs
#SBATCH --partition=shared
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# Resume of f1novel (job 38629780), which TIMED OUT at iter 200/5000 running only
# ~6.7 min/iter -- too slow to reach an informative iteration count. The dominant
# cost was rollout=2048 (2x steps) on top of novelty-driven cache misses, so this
# resume DROPS --rollout to 1024 (~halves per-iter cost; still ~21 episodes/update
# at max_steps=48, plenty). Policy/optimizer are rollout-independent, so resuming
# with a smaller rollout is safe.
#
# CRITICAL: resume restores only policy/optimizer/iters/best; env + PPO config come
# from the CLI, so EVERY shaping flag from the original run is re-passed here
# (--no-stop, --start-mode random, --max-steps 48, --epochs 2, --ent-coef-final
# 0.02, --novelty-coef 0.05, terminal). iters_done=200 loads from the checkpoint,
# so --iters 4800 runs 200 -> 5000. This is a ~1200-iter TEST: watch exact_hits
# going nonzero and best_score climbing off 0.786; if still 0 exact / <=0.786 by
# ~iter 1200, global-novelty-at-0.05 is too diffuse -> concentrate exploration.
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl
sleep $(( RANDOM % 40 ))

uv run python scripts/train.py \
    --iters 4800 --ckpt-every 50 \
    --run-dir runs/f1novel_20260812_122416 --resume latest \
    --metric f1 --normalization physical \
    --start-mode random \
    --no-stop --terminal-mode exact --terminal-bonus 5 \
    --max-steps 48 --rollout 1024 --epochs 2 \
    --ent-coef-final 0.02 --novelty-coef 0.05 \
    --run-name f1novel --seed 0
