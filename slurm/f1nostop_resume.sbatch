#!/bin/bash
#SBATCH --job-name=f1nostop_rs
#SBATCH --partition=intermediate
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# Resume of f1nostop (job 36415083), which TIMED OUT at iter 900/5000 on the
# shared partition's self-imposed 24h. This run is ~23x slower per iter than the
# STOP-enabled runs (no-STOP forces 128-step episodes over novel models -> low
# spectrum-cache hit rate -> ~700 live GroupMath calls/iter, ~1.6 min/iter). The
# full 5000 iters is ~5.6 days, so it runs on `intermediate` (14-day limit) to
# finish in a single job instead of chaining 24h blocks.
#
# CRITICAL: resume restores only policy/optimizer/iters/best from the checkpoint;
# the env config comes from the CLI. So EVERY env-shaping flag from the original
# run must be re-passed here (--no-stop, --start-mode random, terminal settings),
# or the defaults would silently re-enable STOP. iters_done=900 is loaded from
# the checkpoint, so --iters 4100 runs 900 -> 5000 and the ent-coef anneal
# (keyed on iters_done) picks up exactly where it left off.
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# Shared spectrum cache on lab storage (already warmed to ~1.16M entries by the
# first block; hit-rate was climbing as the policy concentrated).
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl

sleep $(( RANDOM % 40 ))

uv run python scripts/train.py \
    --iters 4100 --ckpt-every 50 \
    --run-dir runs/f1nostop_20260730_151652 --resume latest \
    --metric f1 --normalization physical \
    --start-mode random \
    --no-stop --terminal-mode exact --terminal-bonus 5 \
    --run-name f1nostop --seed 0
