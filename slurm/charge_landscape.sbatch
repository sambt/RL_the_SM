#!/bin/bash
#SBATCH --job-name=chargescape
#SBATCH --partition=test
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /n/home11/sambt/iaifi/RL_the_SM
module load mathematica/15.0-fasrc01
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# Warm shared cache (read into a plain dict; the script never writes it back).
export SM_RL_CACHE=/n/holystore01/LABS/iaifi_lab/Users/sambt/sm_rl_cache/spectra.pkl
sleep $(( RANDOM % 20 ))
uv run python scripts/charge_landscape.py
