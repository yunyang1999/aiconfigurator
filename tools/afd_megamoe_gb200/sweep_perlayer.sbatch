#!/usr/bin/env bash
# AIC AFD-vs-colocated sweep with the routing skew measured PER LAYER.
#
# The previous run collapsed all 94 layers into one expert-count vector before
# taking max/mean. A decode step pays `sum over layers of max over ranks`, and
# the busiest rank changes almost every layer, so that collapse let the peaks
# cancel and reported a machine far more balanced than any step actually runs.
# Same formula, same measured curves, same f -- only the imbalance input moves.
#
# Runs as a Slurm job, not on the login node: the login node is x86_64 while
# venv-aic's numpy is built aarch64 (Grace).
# Account and partition come from the environment (SBATCH_ACCOUNT /
# SBATCH_PARTITION) or the command line (sbatch -A ... -p ...).
#SBATCH --job-name=afd-megamoe-sweep
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
cd "${AFD_TOOLDIR:-$SLURM_SUBMIT_DIR}"
PY=${AIC_PY:?set AIC_PY to the venv python -- see .claude/skills/aic-afd-megamoe/SKILL.md}
: "${AFD_MEGAMOE_ROOT:?set AFD_MEGAMOE_ROOT to the unpacked data bundle}"
export PYTHONPATH=../afd_megamoe_b200:.
echo "node=$(hostname) arch=$(uname -m) started=$(date -Is)"
$PY -c "import numpy, pandas; print('numpy', numpy.__version__, 'pandas', pandas.__version__)"

echo
echo "=== pre-check: imbalance and coefficient, per-layer vs layer-summed ==="
$PY - <<'PYEOF'
import os, json
from pathlib import Path
import run_gb200_measured as R
from megamoe_table import MeasuredStage

per = MeasuredStage(R.ROOT, imbalance_dist=R._routing_distributions(),
                    imbalance_measured=R.ROOT / "measured_imbalance_coefficients.json",
                    real_skew=R.ROUTING_IS_REAL)
agg_dist = {}
for m, p in R.ROUTING_DIST_AGGREGATED.items():
    try:
        agg_dist[m] = json.loads(Path(p).read_text())["counts_total"]
    except Exception:
        pass
agg = MeasuredStage(R.ROOT, imbalance_dist=agg_dist,
                    imbalance_measured=R.ROOT / "measured_imbalance_coefficients.json",
                    real_skew=R.ROUTING_IS_REAL)

print(f"{'model':<24}{'experts':>8}{'ranks':>7}{'I layer-summed':>16}{'I per-layer':>13}{'x':>7}")
for m in sorted(per._imbalance_dist):
    ne = len(per._imbalance_dist[m][0])
    nl = len(per._imbalance_dist[m])
    for r in (2, 4, 8, 16, 24, 32):
        if ne % r:
            continue
        ia = agg._rank_imbalance(m, r)
        il = per._rank_imbalance(m, r)
        if ia is None or il is None:
            continue
        k = (il - 1) / (ia - 1) if ia > 1.000001 else float("nan")
        print(f"{m:<24}{ne:>8}{r:>7}{ia:>16.4f}{il:>13.4f}{k:>6.2f}x")
    print(f"{'':<24}{'':>8}({nl} layers kept)")
PYEOF

echo
echo "=== full sweep, per-layer skew ==="
$PY run_gb200_measured.py --workers 48 --out handoff/gb200_perlayer_routing.csv

$PY - <<'PYEOF'
import csv, collections, re, statistics as st
rows = list(csv.DictReader(open("handoff/gb200_perlayer_routing.csv")))
print(f"\nrows {len(rows)}", dict(collections.Counter(r["status"] for r in rows)))
ok = [r for r in rows if r["status"] == "ok"]
c = collections.Counter("routing skew" in (r.get("moe_provenance") or "") for r in ok)
print(f"imbalance-corrected {c[True]}, balanced-only {c[False]}")
pat = re.compile(r"x([\d.]+) \(routing skew")
by = collections.defaultdict(list)
for r in ok:
    m = pat.search(r.get("moe_provenance") or "")
    if m:
        by[r["arm"]].append(float(m.group(1)))
for arm, v in sorted(by.items()):
    s = sorted(v)
    print(f"  {arm:<5} coefficient  median {st.median(v):.4f}  "
          f"p90 {s[9*len(s)//10]:.4f}  max {max(v):.4f}  (n={len(v)})")
PYEOF
echo "finished=$(date -Is)"
