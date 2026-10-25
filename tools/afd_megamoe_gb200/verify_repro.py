"""Re-run a sample of sweep rows and diff them against a stored sweep CSV.

Usage: python verify_repro.py <reference.csv>
"""
import csv, random, sys
from collections import Counter, defaultdict
import run_gb200_measured as R

ref = list(csv.DictReader(open(sys.argv[1])))
rng = random.Random(7)
by = defaultdict(list)
for r in ref:
    by[(r["model"], r["arm"], r["status"])].append(r)

sample = []
for (model, arm, status), rows in sorted(by.items()):
    k = 8 if status == "ok" else 2
    sample += rng.sample(rows, min(k, len(rows)))
# Every colocated EP=24 row: those are the ones that depend on the overlay.
sample += [r for r in ref if r["arm"] == "colo" and r["split"].startswith("ep24")][:20]

def cfg_of(r):
    base = dict(model=r["model"], batch=int(r["batch"]), mtp=int(r["mtp"]),
                isl=int(r["isl"]), osl=int(r["osl"]))
    if r["arm"] == "afd":
        return "afd", dict(base, kind="afd", a=int(r["a"]), f=int(r["f"]),
                           a_tp=int(r["a_tp"]), gran=int(r["gran"]), mb=int(r["mb"] or 1))
    ep = int(r["split"].split("/")[0][2:])
    tp = int(r["a_tp"] or 1)
    return "colo", dict(base, kind="colo", ep=ep, tp=tp)

tally = Counter(); bad = []
for r in sample:
    arm, cfg = cfg_of(r)
    try:
        out = R.run_afd(cfg) if arm == "afd" else R.run_colo(cfg)
    except Exception as e:  # report, do not stop
        out = {"status": f"exception:{type(e).__name__}"}
    s_old, s_new = r["status"], out.get("status")
    if s_old != s_new:
        tally["status_changed"] += 1
        bad.append((r["model"], r["split"], r["batch"], s_old, s_new, "", ""))
        continue
    if s_old != "ok":
        tally["same_non_ok"] += 1
        continue
    t_old, t_new = float(r["tpot_ms"]), float(out["tpot_ms"])
    rel = abs(t_new - t_old) / t_old
    if rel <= 1e-3:
        tally["match"] += 1
    else:
        tally["tpot_changed"] += 1
        bad.append((r["model"], r["split"], r["batch"], s_old, s_new, t_old, t_new))

print(f"\nsampled {len(sample)} rows:", dict(tally))
for b in bad[:40]:
    print("  DIFF", b)
print("\nRESULT:", "PASS" if not bad else "FAIL")
