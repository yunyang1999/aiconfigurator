#!/usr/bin/env bash
# AFD-vs-colocated sweep with the measured MegaMoE stage injected, on
# upstream-main AIC. Default: Kimi-K3 (which only upstream carries) plus
# Qwen3-235B as an anchor on the same engine and data version. MODELS / TAG
# override the model list and output names, e.g. the five existing models
# re-run on upstream's GB200 tables (fp8_block GEMMs were recollected there,
# #1533) for comparison with this branch's CSV.
#
#   engine   AFD_AIC_ROOT -> upstream-main checkout with the measured-stage
#            AFD commits (see that branch's log)
#   systems  make_upstream_systems.sh: these granularity files + gb200_nvl72
#            for the colocated arm, linked to that checkout's data
#   version  sglang 0.5.16 (K3's KDA tables; everything else backward-fills
#            from 0.5.14, for Qwen3 too)
#
# Account and partition come from the environment (SBATCH_ACCOUNT /
# SBATCH_PARTITION) or the command line (sbatch -A ... -p ...).
#SBATCH --job-name=afd-k3-upstream-sweep
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
cd "${AFD_TOOLDIR:-$SLURM_SUBMIT_DIR}"
PY=${AIC_PY:?set AIC_PY to the upstream venv python}
: "${AFD_MEGAMOE_ROOT:?set AFD_MEGAMOE_ROOT to the unpacked data bundle}"
: "${AFD_AIC_ROOT:?set AFD_AIC_ROOT to the upstream AIC checkout}"
export AFD_AIC_SYSTEMS="default,${AFD_MEGAMOE_ROOT}/aic_systems_upstream"
export AFD_COLO_SYSTEM=gb200_nvl72
export AFD_BACKEND_VERSION=0.5.16
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=../afd_megamoe_b200:.
OUT="${AFD_MEGAMOE_ROOT}/handoff"
MODELS=${MODELS:-kimi_k3_fp4 qwen3_235b_fp4}
TAG=${TAG:-k3_upstream}
echo "node=$(hostname) arch=$(uname -m) started=$(date -Is)"
git -C "${AFD_AIC_ROOT}" log --oneline -1

bash make_upstream_systems.sh

echo "=== measured K3 envelope ==="
[[ " ${MODELS} " == *" kimi_k3_fp4 "* ]] && $PY - <<'PYEOF'
import run_gb200_measured as R
T = R.table()
for mtp in (0, 1, 2):
    e = T.envelope(mtp).get("kimi_k3_fp4")
    if not e:
        print(f"nextN={mtp}: no K3 measurements"); continue
    print(f"nextN={mtp}: shards {e['shards']} seqs {e['seqs']} ags {dict(e['ags'])}")
    print(f"          colocated eps {e['eps']} tokens/rank {e['colo_batches']}")
PYEOF

echo "=== sweep ==="
$PY run_gb200_measured.py --models ${MODELS} --workers 64 --out "${OUT}/gb200_measured_${TAG}.csv"

echo "=== capacity walls ==="
$PY precompute_walls.py --workers 64 --csv "${OUT}/gb200_measured_${TAG}.csv" \
    --out "${OUT}/capacity_walls_${TAG}.json"
echo "finished=$(date -Is)"
