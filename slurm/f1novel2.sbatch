#!/bin/bash
#SBATCH --job-name=f1novel2
#SBATCH --partition=sapphire
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# Phase 7: retuned novelty. f1novel FAILED because the bonus owned the objective
# (intrinsic ~70% of return -> policy wandered, best stalled flat at 0.652, 0
# exact). Three complementary fixes, each for a distinct failure:
#   --novelty-coef 0.01         cut early magnitude (target intrinsic <~30% of return)
#   --novelty-mode first_visit  pay once per config, 0 after -> no per-step
#                               accumulation over the 48-step horizon, no camping
#                               reward (unlike count's divergent sum 1/sqrt(n));
#                               self-fades as the space is covered
#   --novelty-anneal-iters 3000 fade the bonus -> 0 so late training exploits and
#                               can sit exactly on an exact match
# Everything else = the f1novel bundle (no-STOP, random start, max_steps 48,
# rollout 1024, epochs 2, ent-floor 0.02). Fresh run (unbiased); run in 24h blocks
# with resume. Watch: ep_return now task-dominated, best_score climbing past 0.786.
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl
sleep $(( RANDOM % 40 ))

uv run python scripts/train.py \
    --iters 5000 --ckpt-every 50 \
    --metric f1 --normalization physical \
    --start-mode random \
    --no-stop --terminal-mode exact --terminal-bonus 5 \
    --max-steps 48 --rollout 1024 --epochs 2 \
    --ent-coef-final 0.02 \
    --novelty-coef 0.01 --novelty-mode first_visit --novelty-anneal-iters 3000 \
    --run-name f1novel2 --seed 0
