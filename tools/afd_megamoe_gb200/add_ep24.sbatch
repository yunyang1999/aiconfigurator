#!/usr/bin/env bash
# Add measured EP=24 MoE rows for DeepSeek-V4-Pro to an OVERLAY copy of the
# performance database.
#
# Why EP=24 was missing at all: every moe_ep_size AIC has ever collected is a
# power of two -- moe_perf.parquet carries [1,2,4,8,16,32,64,128,256] and
# dsv4_megamoe_module_perf.parquet carries [4,8,16,32]. So EP=24 is unpriceable
# for ANY model, not just DeepSeek-V4; V4-Pro is simply the one whose 384
# experts divide by 24, so it is the only model the sweep ever asks about it.
#
# Why an overlay and not an append: data/ is a symlink into installed package
# data that everyone shares. This writes a private tree whose every other file
# is a symlink to the shipped one, so exactly one file differs and the shipped
# database is untouched.
#
# The conversion divides stage_cuda by the MoE layer count, because our
# measurement spans every layer while AIC's latency column is one module. That
# ratio was checked, not assumed: converting our own EP=8 measurements the same
# way reproduces AIC's existing EP=8 rows to within 0.89-1.10, median 0.963.
# Account and partition come from the environment (SBATCH_ACCOUNT /
# SBATCH_PARTITION) or the command line (sbatch -A ... -p ...).
#SBATCH --job-name=afd-megamoe-dsv4-ep24
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -eo pipefail
cd "${AFD_TOOLDIR:-$SLURM_SUBMIT_DIR}"
PY=${AIC_PY:?set AIC_PY to the venv python -- see .claude/skills/aic-afd-megamoe/SKILL.md}
: "${AFD_MEGAMOE_ROOT:?set AFD_MEGAMOE_ROOT to the unpacked data bundle}"
"$PY" - <<'PY'
import glob, json, os, re
import pandas as pd

W = os.environ["AFD_MEGAMOE_ROOT"]
SHIPPED = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                 "aic-core/src/aiconfigurator_core/systems/data")
SRC_FILE = os.path.join(SHIPPED, "gb200/moe/sglang/0.5.10/dsv4_megamoe_module_perf.parquet")
TARGET = (f"{W}/aic_overlay/data/gb200/moe/sglang/0.5.10/"
          "dsv4_megamoe_module_perf.parquet")

ref = pd.read_parquet(SRC_FILE)
print(f"读入原库 {len(ref)} 行(原文件不修改),写出到 overlay")
if (ref.moe_ep_size == 24).any():
    print("EP=24 已存在于原库,无需补")
    raise SystemExit(0)

# Template from the nearest collected point, so every field except the two we
# actually vary is identical to something the loader already accepts.
tmpl = ref[(ref.phase == "generation")
           & (ref.moe_ep_size == 8)
           & (ref.moe_tp_size == 1)].iloc[0].to_dict()

rows = []
for f in sorted(glob.glob(f"{W}/results/megamoe-colocated-ep24/"
                          "deepseek_v4_pro_fp4_24ep_t*_n0_balanced_mega.json")):
    j = json.load(open(f))
    t = int(re.search(r"_t(\d+)_", os.path.basename(f)).group(1))
    layers = j["workload"]["layers"]
    per_layer = j["backend_results"]["mega"]["stage_cuda"]["p50_ms"] / layers
    r = dict(tmpl)
    r.update(num_tokens=t, global_num_tokens=t * 24, moe_ep_size=24,
             latency=round(per_layer, 6))
    nmax = max(512, 1 << max(0, (t * 24 - 1)).bit_length())
    r.update(num_max_tokens_per_rank=nmax,
             effective_num_max_tokens_per_rank=nmax + 64)
    rows.append(r)

if not rows:
    raise SystemExit("no EP=24 measurements found")

add = pd.DataFrame(rows)[list(ref.columns)]
out = pd.concat([ref, add], ignore_index=True)
os.makedirs(os.path.dirname(TARGET), exist_ok=True)
# A file-level symlink here points back into the shipped database, and
# writing through it silently edits AIC's own table -- which is exactly
# what happened once. Replace any link with a real file before writing.
if os.path.islink(TARGET):
    os.unlink(TARGET)
out.to_parquet(TARGET, index=False)
print(f"写入 {len(add)} 行 EP=24 -> overlay 共 {len(out)} 行")
print(add[["num_tokens", "global_num_tokens", "moe_ep_size",
           "latency"]].to_string(index=False))
PY
