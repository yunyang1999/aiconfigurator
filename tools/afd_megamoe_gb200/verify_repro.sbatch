#!/usr/bin/env bash
# Reproducibility check: re-run a sample of sweep rows with the current code and
# data, and compare against the stored sweep CSV.
#
# Run after any change to the database layout, the overlay, or the measured
# tables. Rows that relied on data that is no longer reachable come back with a
# different status or TPOT, which is exactly what this is meant to catch.
# Account and partition come from the environment (SBATCH_ACCOUNT /
# SBATCH_PARTITION) or the command line (sbatch -A ... -p ...).
#SBATCH --job-name=afd-megamoe-verify
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
cd "${AFD_TOOLDIR:-$SLURM_SUBMIT_DIR}"
PY=${AIC_PY:?set AIC_PY to the venv python -- see .claude/skills/aic-afd-megamoe/SKILL.md}
: "${AFD_MEGAMOE_ROOT:?set AFD_MEGAMOE_ROOT to the unpacked data bundle}"
export PYTHONPATH=../afd_megamoe_b200:.
REF=${AFD_VERIFY_CSV:-$AFD_MEGAMOE_ROOT/reference/gb200_perlayer_routing.csv}
echo "node=$(hostname) arch=$(uname -m) ref=$REF started=$(date -Is)"
$PY verify_repro.py "$REF"
echo "finished=$(date -Is)"
