#!/bin/bash
#SBATCH --job-name=f1novel
#SBATCH --partition=shared
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM

# The "explore the charge subspace" bundle, motivated by scripts/charge_landscape.py:
# the f1nostop 0.786 plateau is structurally the SM (SU3, 3 fundamental fermions)
# and the diagnostic proved it's an EXPLORATION FAILURE, not a landscape trap --
# an exact match sits a short charge-lattice walk away, behind a shallow (~0.08)
# reward valley that a collapsed-entropy policy won't cross. This run attacks that:
#   --novelty-coef 0.05  count-based bonus keyed on physics (canonical_key); near
#                        the plateau, "new physics" == new charge configs, and it
#                        can pay the temporary score cost of crossing the valley.
#   --ent-coef-final 0.02 keep entropy alive (was bleeding to 0.005 -> premature
#                        convergence exactly when it found 0.786).
#   --max-steps 48       shorter episodes: ~42 episodes/2048-step update instead of
#                        f1nostop's 8 -> far lower-variance gradients, less overfit.
#   --epochs 2           gentler PPO updates (was 4) -> slower entropy collapse.
#   --rollout 2048       bigger batch to go with the above.
# Still no-STOP + random start (episodes end at max_steps or an exact match).
#
# RUNTIME: novelty + no-STOP both push toward NOVEL models -> more spectrum-cache
# misses -> slow (~2-3 min/iter at 2048 steps/iter). 5000 iters won't finish in
# one 24h block, so this runs on `shared` (short request => backfills fast, unlike
# a multi-day `intermediate` request that just sits in queue) and RESUMES across
# blocks via `--run-dir ... --resume latest` (ckpt-every 50). The shared cache is
# already ~1.35M entries and keeps growing, so later blocks should run faster.
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl
sleep $(( RANDOM % 40 ))

uv run python scripts/train.py \
    --iters 5000 --ckpt-every 50 \
    --metric f1 --normalization physical \
    --start-mode random \
    --no-stop --terminal-mode exact --terminal-bonus 5 \
    --max-steps 48 --rollout 2048 --epochs 2 \
    --ent-coef-final 0.02 --novelty-coef 0.05 \
    --run-name f1novel --seed 0
