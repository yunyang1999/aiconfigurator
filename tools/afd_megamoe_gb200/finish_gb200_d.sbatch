#!/usr/bin/env bash
# Stage 4: recompute the colocated walls that a missing argument had silently
# turned into blanks, then rebuild.
#
# _colo_memory called cli_estimate with nextn but without nextn_accepted. AIC
# needs both, the call raised, and colo_wall's except clause reported that as
# "no wall". Every one of the 184 unresolved walls was colocated and nextN 1 or
# 2 -- which is where AFD wins most often, so the rows that most needed the
# ladder check were exactly the ones missing it. The bare except on the first
# probe is gone too, so a future failure surfaces instead of emptying a column.
#SBATCH --job-name=mm-gb200-fixwalls
#SBATCH --time=02:00:00
#SBATCH --output=/home/scratch.yuny_wwfo_1/megamoe/logs/fixwalls_%j.out
#SBATCH --error=/home/scratch.yuny_wwfo_1/megamoe/logs/fixwalls_%j.err
set -euo pipefail
ROOT=/home/scratch.yuny_wwfo_1/megamoe
PY=/home/scratch.yuny_wwfo_1/aic/venv-aic/bin/python
cd "${ROOT}"
echo "node=$(hostname) started=$(date -Is)"
$PY precompute_walls.py --workers 14
$PY - <<'PYEOF'
import json
from pathlib import Path
c = json.loads(Path("handoff/capacity_walls.json").read_text())
bad = {k: v for k, v in c.items() if not isinstance(v, list)}
print(f"walls {len(c)}, resolved {len(c)-len(bad)}, unresolved {len(bad)}")
for k, v in list(bad.items())[:5]:
    print(f"   {k} -> {v}")
PYEOF
$PY build_gb200_style.py
cp handoff/AIC_AFD_MegaMoE_GB200_NVL72.xlsx "${AFD_DELIVER_DIR:?set AFD_DELIVER_DIR to the directory that receives the workbook}/"
$PY - <<'PYEOF'
from openpyxl import load_workbook
import collections
import os
wb = load_workbook(os.path.join(os.environ["AFD_DELIVER_DIR"], "AIC_AFD_MegaMoE_GB200_NVL72.xlsx"), read_only=True)
ws = wb["Summary"]; rows = list(ws.iter_rows(min_row=3, values_only=True))
h = list(rows[0]); i = {x: k for k, x in enumerate(h) if x}
b = [r for r in rows[1:] if r and r[0]]
have = sum(1 for r in b if r[i["rung_inflation"]] is not None)
print(f"Summary {len(b)} rows, rung_inflation present on {have}")
print(dict(collections.Counter(r[i["verdict"]] for r in b)))
gain = [r for r in b if r[i["verdict"]] == "gain"]
miss = sum(1 for r in gain if r[i["rung_inflation"]] is None)
print(f"gain rows {len(gain)}, of which still without a ladder check: {miss}")
PYEOF
echo "finished=$(date -Is)"
