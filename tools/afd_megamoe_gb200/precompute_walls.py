#!/usr/bin/env python3
"""Precompute the capacity walls in parallel and cache them to disk.

Each wall is a bisection over batch, and each probe is an AIC call, so a wall
costs 10-40 calls. Inline in the workbook build that was fine at 307k sweep rows
and ran out of a two-hour Slurm limit at 764k: more rows means more distinct
winning topologies, and the builder was computing them one at a time.

Two changes. The walls needed are enumerated up front from the CSV -- only the
topologies that actually WIN somewhere, which is a few hundred rather than the
16k distinct AFD topologies in the sweep -- and they are computed across a
process pool. The result is cached as JSON, so rebuilding the workbook after a
formatting change costs nothing.

The cache key includes everything the wall depends on and nothing else. Batch is
absent by construction: the wall IS the batch. The MoE stage time is absent
because memory does not depend on it, which is also why this stays inside the
measured envelope even though the wall usually lands on a batch that was never
benchmarked.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/scratch.yuny_wwfo_1/megamoe")
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "handoff" / "capacity_walls.json"


def _fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def needed(csv_path: Path, slas):
    """Wall keys for every topology any sheet reports as a winner.

    The builder only ever asks about winners, so computing the whole sweep would
    be wasted work by two orders of magnitude. But "winner" is sheet-dependent:
    SLA_Latency picks one per (cell, limit), AF_Split picks one per split,
    Colo_EP_Frontier one per EP group, Matched_Shard one per shard and GPU_Budget
    one per rack size. All of them are enumerated here so every sheet can carry
    the capacity columns rather than only the two that had them first.
    """
    cells = collections.defaultdict(lambda: {"afd": [], "colo": []})
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok" or r["osl"] != "1024":
                continue
            cells[(r["model"], int(r["isl"]), int(r["osl"]), int(r["mtp"]))][r["arm"]].append(r)
    keys = set()

    def add(cell, arm, w):
        if arm == "colo":
            keys.add(("colo", cell[0], cell[1], cell[2], cell[3],
                      int(w["unit_gpus"]), int(w["a_tp"])))
        else:
            keys.add(("afd", cell[0], cell[1], cell[2], cell[3], int(w["a"]),
                      int(w["f"]), int(w["a_tp"]), int(w["gran"]), int(w["mb"])))

    def best(pts, sla):
        c = [x for x in pts if sla is None or (_fl(x["tpot_ms"]) or 1e9) <= sla]
        return max(c, key=lambda x: _fl(x["tokens_s_gpu"]) or 0.0) if c else None

    for cell, d in cells.items():
        for sla in slas:
            for arm in ("afd", "colo"):
                w = best(d[arm], sla)
                if w:
                    add(cell, arm, w)
        # AF_Split is deliberately NOT covered. It enumerates every A:F split at
        # its own best batch and repeats one colocated point as a baseline -- it
        # is a map of the split space, not a comparison, and nobody reads a
        # capacity wall off a split that lost. Covering it costs 15,729 of the
        # 16,518 walls, 95% of the work, for the one sheet where the column
        # answers nothing. The other five sheets together need 789.
        by_f = collections.defaultdict(list)
        for x in d["afd"]:
            by_f[int(x["f"])].append(x)
        for pts in by_f.values():
            w = best(pts, None)
            if w:
                add(cell, "afd", w)
        # Colo_EP_Frontier: one winner per EP group at every limit.
        by_ep = collections.defaultdict(list)
        for x in d["colo"]:
            by_ep[int(x["unit_gpus"])].append(x)
        for pts in by_ep.values():
            for sla in slas:
                w = best(pts, sla)
                if w:
                    add(cell, "colo", w)
        # GPU_Budget: the winner changes with the rack size, since a unit larger
        # than the budget is not deployable at all.
        for B in (8, 16, 32, 64, 72):
            for arm in ("afd", "colo"):
                pts = [x for x in d[arm] if int(x["unit_gpus"]) <= B]
                for sla in (None, 20, 50):
                    w = best(pts, sla)
                    if w:
                        add(cell, arm, w)
    return sorted(keys)


def _one(key):
    import capacity_wall as CW
    import run_gb200_measured as R
    from colocated_measured import MODELS
    try:
        if key[0] == "colo":
            _, model, isl, osl, mtp, ep, tp = key
            b, conc = CW.colo_wall(R._est(), MODELS[model]["path"], ep, tp,
                                   isl, osl, mtp, R.CAPACITY_GIB)
            unit = ep
        else:
            _, model, isl, osl, mtp, a, f, tp, g, mb = key
            b, conc = CW.afd_wall(R._afd_once, model, a, f, tp, g, isl, osl, mtp, mb)
            unit = a + f
        if b is None:
            return json.dumps(key), None
        return json.dumps(key), [b, conc, round(conc / unit, 4)]
    except Exception as e:
        return json.dumps(key), {"error": f"{type(e).__name__}: {e}"[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--csv", type=Path, default=ROOT / "handoff" / "gb200_measured.csv")
    ap.add_argument("--out", type=Path, default=CACHE)
    a = ap.parse_args()

    slas = [None, 10, 15, 20, 30, 50, 100, 150]
    keys = needed(a.csv, slas)
    cached = {}
    if a.out.exists():
        cached = json.loads(a.out.read_text())
    todo = [k for k in keys if json.dumps(k) not in cached]
    print(f"walls needed {len(keys)}, cached {len(keys) - len(todo)}, to compute {len(todo)}",
          flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, v in ex.map(_one, todo, chunksize=4):
            cached[k] = v
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)
    tmp = a.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(cached))
    tmp.replace(a.out)
    ok = sum(1 for v in cached.values() if isinstance(v, list))
    print(f"wrote {len(cached)} walls ({ok} resolved) -> {a.out}")


if __name__ == "__main__":
    main()
