#!/usr/bin/env bash
# Fill moe_perf's EP=24 hole by interpolating AIC's OWN EP=16 and EP=32 rows.
#
# Why this is the right filler rather than our megamoe measurement: the sweep
# SUBSTITUTES the measured MegaMoE stage for AIC's MoE ops, so whatever AIC
# computes for MoE is discarded. What blocks the row is simply that the estimate
# cannot complete. Injecting a different kernel's latencies into a generic table
# would be inventing data; interpolating the table's own neighbours in EP keeps
# it internally consistent and is visible as such.
#
# Every moe_ep_size AIC ships is a power of two, so EP=24 is unpriceable for any
# model. DeepSeek-V4-Pro is the only one the sweep asks about it, because 384
# experts divide by 24.
# Account and partition come from the environment (SBATCH_ACCOUNT /
# SBATCH_PARTITION) or the command line (sbatch -A ... -p ...).
#SBATCH --job-name=afd-megamoe-moeperf-ep24
#SBATCH --nodes=1
#SBATCH --time=00:25:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
cd "${AFD_TOOLDIR:-$SLURM_SUBMIT_DIR}"
PY=${AIC_PY:?set AIC_PY to the venv python -- see .claude/skills/aic-afd-megamoe/SKILL.md}
: "${AFD_MEGAMOE_ROOT:?set AFD_MEGAMOE_ROOT to the unpacked data bundle}"
"$PY" - <<'PY'
import os
import pandas as pd

OV = os.path.join(os.environ["AFD_MEGAMOE_ROOT"], "aic_overlay/data")
SHIPPED = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                 "aic-core/src/aiconfigurator_core/systems/data")
SRC = os.path.join(SHIPPED, "gb200/moe/sglang/0.5.14/moe_perf.parquet")
TARGET = f"{OV}/gb200/moe/sglang/0.5.14/moe_perf.parquet"

d = pd.read_parquet(SRC)
print(f"原表 {len(d)} 行, ep sizes {sorted(d.moe_ep_size.unique())}")
if (d.moe_ep_size == 24).any():
    print("EP=24 已存在"); raise SystemExit(0)

key = [c for c in d.columns if c not in ("moe_ep_size", "latency")]
# The table has repeated measurements per configuration, so the key is not
# unique; collapse to the median before pairing rather than trying to align
# duplicate rows against each other.
lo = d[d.moe_ep_size == 16].groupby(key, dropna=False)["latency"].median()
hi = d[d.moe_ep_size == 32].groupby(key, dropna=False)["latency"].median()
common = lo.index.intersection(hi.index)
print(f"EP=16 {len(lo)} 组, EP=32 {len(hi)} 组, 可配对 {len(common)}")
if not len(common):
    raise SystemExit("no matched EP=16/32 pairs to interpolate")

# linear in ep between 16 and 32; 24 sits exactly halfway
w = (24 - 16) / (32 - 16)
add = (lo.loc[common] * (1 - w) + hi.loc[common] * w).rename("latency").reset_index()
add["moe_ep_size"] = 24
add = add[list(d.columns)]

out = pd.concat([d, add], ignore_index=True)
os.makedirs(os.path.dirname(TARGET), exist_ok=True)
# A file-level symlink here points back into the shipped database, and
# writing through it silently edits AIC's own table -- which is exactly
# what happened once. Replace any link with a real file before writing.
if os.path.islink(TARGET):
    os.unlink(TARGET)
out.to_parquet(TARGET, index=False)
print(f"写入 {len(add)} 行 EP=24 -> overlay 共 {len(out)} 行")

chk = add[(add.num_experts == 384) & (add.hidden_size == 7168)]
print(f"\n384专家/hidden7168 的 EP=24 行: {len(chk)}")
if len(chk):
    cols = [c for c in ("num_tokens", "topk", "moe_tp_size", "latency") if c in chk.columns]
    print(chk.sort_values("num_tokens")[cols].head(12).to_string(index=False))
PY
