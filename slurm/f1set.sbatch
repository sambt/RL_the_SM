#!/bin/bash
#SBATCH --job-name=f1set
#SBATCH --partition=sapphire
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# Phase 9: structural fix to the ACTION SPACE (optimizer-side levers exhausted,
# Phases 6-8). Two changes, nothing else:
#   --charge-actions set  one action per lattice value per charge (64 actions):
#                         any charge value is ONE move away, so the multi-unit
#                         charge fixes the Phase-5 diagnostic found are no longer
#                         valleys of scored intermediates. The lattice itself
#                         (charge_max=6, steps 1/1/1/0.5) is untouched -- NOT
#                         tightened toward SM values (that would be SM prior).
#   --cursor              CURSOR_PREV/NEXT so edits reach ANY parton. Without it
#                         every edit hits the last parton, so fixing parton 1 of
#                         3 meant REMOVE x2 -> edit -> rebuild x2 (found 09-02;
#                         this is why even the gentle warm start never crossed).
# Head: 18 -> 76 actions. Exact SM reachable in 19 actions from empty SU3 (mock).
# Fresh run (no warm start: head shape changed; and unbiased). Otherwise the
# f1nostop recipe that is known to concentrate: epochs 4, ent 0.03->0.005/5000,
# no novelty; with the Phase-7 horizon (max_steps 48, rollout 1024).
# Watch: best_score past 0.786, exact hits; entropy (max now ln~70 = 4.2, not 2.8)
# falling below ~2 by iter ~1000 as f1nostop's did.
#
# 2026-09-08 resubmission. Attempt 1 (job 44389130) was OOM-killed at iter 38 on
# 24G: the charge-keyed spectrum cache (2.5M entries, 5 GB pickle, ~20 GB in RAM)
# got ~800 novel entries/iter under set-charge and re-loaded the whole pickle on
# every 200-entry flush (4x/iter, 2-3x RAM spikes, and most of the 5 min/iter).
# Fix: the expensive GetSpins call is charge-free, so its memo is now
# canonicalized + persisted (SM_RL_SPINS_MEMO, ~KB) and the big spectra cache is
# NOT loaded (in-memory, bounded). A spectrum miss now costs only Python.
# Validated vs the real engine: scripts/validate_spins_memo.py.
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
unset SM_RL_CACHE
export SM_RL_SPINS_MEMO=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spins_memo.pkl
sleep $(( RANDOM % 40 ))

uv run python scripts/train.py \
    --iters 5000 --ckpt-every 25 \
    --metric f1 --normalization physical \
    --start-mode random \
    --no-stop --terminal-mode exact --terminal-bonus 5 \
    --max-steps 48 --rollout 1024 \
    --charge-actions set --cursor \
    --run-name f1set --seed 0
