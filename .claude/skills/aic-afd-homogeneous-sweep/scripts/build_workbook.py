"""Reduce a sweep CSV into a result workbook.

Column layout follows the reference workbooks: identity, then the constraint
in force on that sheet, then the constant descriptors, then the AFD side, the
colocated side and the ratio.  Identity and constraint stay frozen so they
remain visible when scrolling right.  Rows whose ratio exceeds 1.00 are filled
green.

Usage: build_wb.py <csv> <out.xlsx> <title> <multi|m3> [fixed|search]
"""
import sys

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

CSV, OUT, TITLE, BOOK = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
MODE = sys.argv[5] if len(sys.argv) > 5 else "fixed"

GREEN = PatternFill("solid", fgColor="C6EFCE")
HDR = PatternFill("solid", fgColor="1F3864")
HDR_FONT = Font(color="FFFFFF", bold=True, size=10)
NOTE_FONT = Font(italic=True, size=9, color="595959")
SLA_MS = [10, 15, 20, 30, 50, 100, 150]
SPEED_TARGETS = [10, 20, 30, 40, 50, 60, 70, 100]   # tokens/s/user
HBM_GB = 185.0

d = pd.read_csv(CSV)
d = d[d.tokens_s_gpu.notna()]
d["attn"] = d["attn"].replace({"MSA (linear)": "MSA"})
KEY = ["model", "attn", "precision", "isl", "osl", "mtp_nextn"]
BATCH_MAX = int(d.batch_per_unit.max())


def best(g, sla=None, speed=None):
    """Highest tokens/s/gpu in g, optionally under a latency or speed floor."""
    if sla is not None:
        g = g[g.tpot_ms <= sla]
    if speed is not None:
        g = g[g.tokens_s_user >= speed]
    return None if not len(g) else g.loc[g.tokens_s_gpu.idxmax()]


def limited_by(g, row, sla=None, speed=None):
    """What stopped the winning batch from growing.

    ``capacity`` -- no larger batch fits in HBM.
    ``latency``  -- larger batches fit, but violate the constraint in force on
                    this sheet. Without separating this from ``compute`` a
                    tightened limit silently reads as saturated compute.
    ``compute``  -- larger batches both fit and satisfy the constraint, yet do
                    not raise throughput.
    ``grid``     -- the batch grid ran out of rungs before anything bound.
    """
    same = g[g.config == row.config]
    if not len(same):
        return ""
    top_feasible = int(same.batch_per_unit.max())
    ok = same
    if sla is not None:
        ok = ok[ok.tpot_ms <= sla]
    if speed is not None:
        ok = ok[ok.tokens_s_user >= speed]
    top_allowed = int(ok.batch_per_unit.max()) if len(ok) else top_feasible
    if int(row.batch_per_unit) < top_allowed:
        return "compute"
    if top_allowed < top_feasible:
        return "latency"
    return "capacity" if top_feasible < BATCH_MAX else "grid"


def weights_gb(g, cfg):
    """Weight footprint per GPU, read off the smallest batch of that config."""
    same = g[g.config == cfg]
    if not len(same):
        return None
    lo = same.loc[same.batch_per_unit.idxmin()]
    return round(float(lo.memory_gb), 1)


ID = ["model", "attn", "precision", "isl", "osl", "mtp_nextn"]
DESC = ["moe_kernel", "system", "backend"]


def order(df_out, constraint=()):
    lead = ID + list(constraint) + DESC
    rest = [c for c in df_out.columns if c not in lead]
    return df_out[[c for c in lead if c in df_out.columns] + rest]


def pair_rows(df, sla=None, speed=None, extra=()):
    out = []
    for key, g in df.groupby(KEY + list(extra)):
        afd, colo = g[g.arm == "afd"], g[g.arm == "colocated"]
        a, c = best(afd, sla, speed), best(colo, sla, speed)
        if a is None and c is None:
            continue
        row = dict(zip(KEY + list(extra), key))
        row["moe_kernel"] = g.moe_kernel.iloc[0]
        row["system"] = g.system.iloc[0]
        row["backend"] = g.backend.iloc[0]
        if a is not None:
            conc = float(a.concurrency) if pd.notna(a.concurrency) else float("nan")
            row.update(
                afd_split=a.config, afd_A_gpus=int(a.a_gpus), afd_F_gpus=int(a.f_gpus),
                afd_unit_gpus=int(a.unit_gpus), afd_batch=int(a.batch_per_unit),
                afd_tpsgpu=round(a.tokens_s_gpu, 1), afd_tpot_ms=round(a.tpot_ms, 2),
                afd_tokps_per_user=round(a.tokens_s_user, 2),
                afd_conc=None if pd.isna(conc) else int(conc),
                afd_conc_per_gpu=None if pd.isna(conc) else round(conc / a.unit_gpus, 1),
                afd_t_attn_us=a.t_attn_us, afd_t_ffn_us=a.t_ffn_us,
                afd_t_comm_us=a.t_comm_us, afd_tf_over_ta=a.tf_over_ta,
                afd_comm_hidden=a.comm_hidden,
                afd_memory_gb=round(a.memory_gb, 1),
                afd_batch_limited_by=limited_by(afd, a, sla, speed))
        if c is not None:
            conc = float(c.concurrency) if pd.notna(c.concurrency) else float("nan")
            w = weights_gb(colo, c.config)
            row.update(
                colo_cfg=c.config, colo_unit_gpus=int(c.unit_gpus),
                colo_batch=int(c.batch_per_unit), colo_tpsgpu=round(c.tokens_s_gpu, 1),
                colo_tpot_ms=round(c.tpot_ms, 2),
                colo_tokps_per_user=round(c.tokens_s_user, 2),
                colo_conc=None if pd.isna(conc) else int(conc),
                colo_conc_per_gpu=None if pd.isna(conc) else round(conc / c.unit_gpus, 1),
                colo_memory_gb=round(c.memory_gb, 1),
                colo_weights_per_gpu_GB=w,
                colo_weight_share_of_hbm_pct=None if w is None else round(100 * w / HBM_GB, 1),
                colo_batch_limited_by=limited_by(colo, c, sla, speed))
        if a is not None and c is not None and c.tokens_s_gpu > 0:
            row["ratio"] = round(a.tokens_s_gpu / c.tokens_s_gpu, 3)
        out.append(row)
    if not out:
        return pd.DataFrame()
    return order(pd.DataFrame(out), extra)


def write(ws, df, note, n_freeze=6):
    ws["A1"] = note
    ws["A1"].font = NOTE_FONT
    if df is None or not len(df):
        ws["A3"] = "No feasible configuration under the stated constraints."
        return
    cols = list(df.columns)
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=2, column=j, value=c)
        cell.fill, cell.font = HDR, HDR_FONT
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    ri = "ratio" in cols
    for i, (_, r) in enumerate(df.iterrows(), 3):
        green = ri and pd.notna(r.get("ratio")) and r["ratio"] > 1.0
        for j, c in enumerate(cols, 1):
            v = r[c]
            cell = ws.cell(row=i, column=j, value=None if pd.isna(v) else v)
            if green:
                cell.fill = GREEN
    for j, c in enumerate(cols, 1):
        w = max([len(str(c))] + [len(str(v)) for v in df[c].head(300)])
        ws.column_dimensions[get_column_letter(j)].width = min(max(w + 2, 9), 24)
    ws.freeze_panes = f"{get_column_letter(n_freeze + 1)}3"


wb = Workbook()

# ---------------------------- Configuration ----------------------------
ws = wb.active
ws.title = "Configuration"
rows = [
    ("Item", "Detail"),
    ("Subject", "Homogeneous GPU-to-GPU Attention-FFN Disaggregation versus colocated "
                "serving, decode phase."),
    ("Hardware", "NVIDIA GB200 NVL72. 4 GPUs per node, 72 GPUs per rack, 185 GB usable "
                 "HBM per GPU."),
    ("Serving framework", f"SGLang {d.backend.iloc[0].split()[-1]} packaged performance tables."),
    ("Estimation engine", "AIConfigurator, AFD serving mode. " + (
        "Every deployment and batch is enumerated explicitly and estimated one at a time."
        if MODE == "fixed" else
        "The engine performs its own configuration search and returns the feasible frontier.")),
    ("Database mode", ", ".join(sorted(d.database_mode.unique()))),
    ("Models", ", ".join(sorted(d.model.unique()))),
    ("Attention", "; ".join(f"{m}: {a}" for m, a in
                            sorted(d[["model", "attn"]].drop_duplicates().values.tolist()))),
    ("precision (column)", "The quantisation of the MoE expert weights, and the only "
                           "precision varied in this sweep. Attention, KV cache, the "
                           "projection GEMMs and the collectives keep each model's own "
                           "defaults and do not change with this column. "
                           + ", ".join(sorted(d.precision.unique())) + "."),
    ("moe_kernel (column)", "The kernel that follows from the precision column: FP8-block "
                            "to the Triton fused MoE, NVFP4 to the FlashInfer CuteDSL "
                            "grouped GEMM, INT4-wo to the Marlin MoE."),
    ("MoE kernel", ", ".join(sorted(d.moe_kernel.unique()))),
] + ([
    ("MoE pricing", "The MoE is priced in NVFP4 with the FlashInfer CuteDSL grouped-GEMM "
                    "kernel. The packaged tables hold no row collected at this model's MoE "
                    "shape (hidden 6144, intermediate 3072, 128 experts, top-4), so the "
                    "nearest collected configuration at the SAME quantisation is used and "
                    "rescaled to this shape. Quantisation is never substituted."),
    ("Attention pricing", "Linear (state) attention has no collected data in any packaged "
                          "table. It is priced as its speed-of-light time divided by the "
                          "measured utilisation of DeepSeek sparse attention at the same "
                          "batch, sequence length and prefix, times a scale factor held at "
                          "1.0. The speed-of-light term supplies the shape trend across "
                          "batch and sequence length; the borrowed utilisation supplies the "
                          "absolute efficiency level. The speed-of-light model splits the "
                          "layer the same way as sparse attention does: projection GEMMs, "
                          "FP8 indexer, and the attention kernel itself."),
    ("Other operators", "Projection and shared-expert GEMMs, normalisation, and all "
                        "collective communication are read from collected rows for this "
                        "GPU, framework version and quantisation; no substitution."),
] if BOOK == "m3" else []) + [
    ("Input length", ", ".join(str(v) for v in sorted(d.isl.unique()))),
    ("Generation length", ", ".join(str(v) for v in sorted(d.osl.unique()))),
    ("Speculative decoding", "Draft lengths " + ", ".join(sorted(d.mtp_nextn.astype(str).unique()))
     + ". Conditional acceptance falls with draft position (0.85, 0.80), so draft length 1 "
       "accepts 0.85 tokens per step on average and draft length 2 accepts 1.53."),
    ("Colocated search", "Attention data-parallel and tensor-parallel crossed with MoE "
                         "expert-parallel and tensor-parallel. Unit sizes 2/4/8/16/32/64 GPU."),
    ("AFD search", "Every A:F split that fits one rack at node granularity, up to 17:1, A-pool "
                   "tensor parallelism 1/2/4, F-pool expert-parallel across the whole F pool, "
                   "2 micro-batches, conservative pipeline overlap."),
    ("Batch selection", "Geometric grid stepping near x1.15 through the mid range rather than "
                        "doubling, so a capacity-limited point can land close to the HBM limit "
                        "instead of overshooting it. 53 rungs from 1 to 4096."
     if MODE == "fixed" else
     "Solved analytically rather than enumerated: memory is sampled at two batch sizes to "
     "recover the linear model, then the largest batch satisfying both the HBM capacity and "
     "the KV-cache budget is taken, aligned down to a multiple of 8. A capacity-limited point "
     "therefore lands on the memory limit exactly."),
    ("Evaluation rule", "Each arm is swept over its own configuration space; points exceeding "
                        "the HBM limit are discarded; the highest tokens/s/GPU among the "
                        "survivors represents that arm. Where a latency limit or a per-user "
                        "speed floor applies, points violating it are discarded first."),
    ("Throughput metric", "Decode tokens per second divided by every GPU in the deployment, "
                          "including AFD FFN GPUs that host no requests."),
    ("Ratio", "AFD tokens/s/GPU divided by colocated tokens/s/GPU at the same model, precision, "
              "context, generation length, draft length and constraint. Rows above 1.00 are "
              "shaded."),
    ("batch_limited_by", "What stopped the winning batch from growing. capacity: no larger batch "
                         "fits in HBM. latency: larger batches fit but breach the limit in force on "
                         "that sheet. compute: larger batches both fit and meet the limit, yet do "
                         "not raise throughput. grid: the batch grid ran out of rungs."),
    ("Latency limits", ", ".join(f"{v} ms" for v in SLA_MS) + " time per output token."),
    ("Configuration points", f"{len(d):,}"),
    ("Validation status", "Model output. No AFD runtime measurement is available; no figure "
                          "has been confirmed against hardware."),
]
for i, (a, b) in enumerate(rows, 1):
    ws.cell(row=i, column=1, value=a)
    ws.cell(row=i, column=2, value=b)
    if i == 1:
        for j in (1, 2):
            ws.cell(row=i, column=j).fill = HDR
            ws.cell(row=i, column=j).font = HDR_FONT
ws.column_dimensions["A"].width = 24
ws.column_dimensions["B"].width = 120
for r in range(2, len(rows) + 1):
    ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")

# ---------------------------- result sheets ----------------------------
if BOOK == "multi":
    write(wb.create_sheet("Model_x_Precision"),
          pair_rows(d[(d.osl == 1024) & (d.mtp_nextn.astype(str) == "0")]),
          "Model and weight precision at generation length 1024, draft length 0, "
          "no latency limit. Each arm at its own best deployment and batch.")

write(wb.create_sheet("Context"), pair_rows(d[d.osl == 1024]),
      "Context sweep at generation length 1024, no latency limit. "
      "Each arm at its own best deployment and batch.")

write(wb.create_sheet("Context_x_Generation" if BOOK == "m3" else "ISL_x_OSL"),
      pair_rows(d[d.mtp_nextn.astype(str) == "0"]),
      "Input length crossed with generation length, draft length 0, no latency limit.")

if BOOK == "m3":
    tgt = []
    for s in SPEED_TARGETS:
        t = pair_rows(d[d.osl == 1024], speed=s)
        if len(t):
            t.insert(len(KEY), "target_tokps_per_user", s)
            t.insert(len(KEY) + 1, "equiv_TPOT_ms", round(1000.0 / s, 2))
            tgt.append(t)
    write(wb.create_sheet("Speed_Targets"),
          pd.concat(tgt, ignore_index=True) if tgt else None,
          "Best achievable per-GPU throughput at each per-user speed floor, generation "
          "length 1024. A blank side means no configuration of that arm reaches the floor.",
          n_freeze=8)

lat = []
for sla in SLA_MS:
    t = pair_rows(d[d.osl == 1024], sla=sla)
    if len(t):
        t.insert(len(KEY), "sla_tpot_ms", sla)
        lat.append(t)
write(wb.create_sheet("Latency"), pd.concat(lat, ignore_index=True) if lat else None,
      "Best configuration under each time-per-output-token limit, generation length 1024. "
      "A blank side means no configuration of that arm meets the limit.", n_freeze=7)

def af_split_rows(df):
    """One row per A:F split, each against the single best colocated point."""
    out = []
    for key, g in df.groupby(KEY):
        c = best(g[g.arm == "colocated"])
        colo = g[g.arm == "colocated"]
        for cfg, ga in g[g.arm == "afd"].groupby("config"):
            a = best(ga)
            if a is None:
                continue
            conc = float(a.concurrency) if pd.notna(a.concurrency) else float("nan")
            row = dict(zip(KEY, key))
            row.update(moe_kernel=g.moe_kernel.iloc[0], system=g.system.iloc[0],
                       backend=g.backend.iloc[0], afd_split=cfg,
                       afd_A_gpus=int(a.a_gpus), afd_F_gpus=int(a.f_gpus),
                       afd_unit_gpus=int(a.unit_gpus), afd_batch=int(a.batch_per_unit),
                       afd_tpsgpu=round(a.tokens_s_gpu, 1), afd_tpot_ms=round(a.tpot_ms, 2),
                       afd_tokps_per_user=round(a.tokens_s_user, 2),
                       afd_conc=None if pd.isna(conc) else int(conc),
                       afd_conc_per_gpu=None if pd.isna(conc) else round(conc / a.unit_gpus, 1),
                       afd_t_attn_us=a.t_attn_us, afd_t_ffn_us=a.t_ffn_us,
                       afd_t_comm_us=a.t_comm_us, afd_tf_over_ta=a.tf_over_ta,
                       afd_comm_hidden=a.comm_hidden, afd_memory_gb=round(a.memory_gb, 1),
                       afd_batch_limited_by=limited_by(ga, a))
            if c is not None and c.tokens_s_gpu > 0:
                w = weights_gb(colo, c.config)
                row.update(colo_cfg=c.config, colo_unit_gpus=int(c.unit_gpus),
                           colo_batch=int(c.batch_per_unit),
                           colo_tpsgpu=round(c.tokens_s_gpu, 1),
                           colo_tpot_ms=round(c.tpot_ms, 2),
                           colo_weights_per_gpu_GB=w,
                           ratio=round(a.tokens_s_gpu / c.tokens_s_gpu, 3))
            out.append(row)
    if not out:
        return pd.DataFrame()
    r = order(pd.DataFrame(out))
    return r.sort_values(KEY + ["afd_tpsgpu"], ascending=[True] * len(KEY) + [False])


write(wb.create_sheet("AF_Split"),
      af_split_rows(d[(d.osl == 1024) & (d.mtp_nextn.astype(str) == "0")]),
      "Every A:F split at its own best batch, generation length 1024, draft length 0, "
      "no latency limit. The colocated side repeats the single best colocated point.")

wb.properties.title = TITLE
wb.save(OUT)
print(f"{OUT}\n   sheets={wb.sheetnames}")
