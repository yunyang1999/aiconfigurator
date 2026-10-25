#!/usr/bin/env bash
# Build the AIC systems overlay this sweep reads, from THIS checkout's shipped
# database, and point tools/afd_megamoe_gb200/data at it.
#
# Why it exists: AIC ships moe_ep_size only in powers of two, so DeepSeek-V4-Pro
# at EP=24 cannot be priced. The two add_*.sbatch builders write EP=24 rows, and
# they must write into a private copy, never into the shipped tables.
#
# Why a symlink mirror and not a copy: the shipped tree is large and the overlay
# must differ from it in exactly two files. `cp -as` makes real directories and
# one symlink per file; each builder unlinks its target before writing, so only
# that one file becomes real. Two things this layout prevents, both seen once:
#   - a directory-level symlink lets AIC's version-reuse resolution walk back
#     into the shipped tree, so the overlay is silently ignored;
#   - writing to a file-level symlink edits the shipped table it points at.
#
# Why tools/afd_megamoe_gb200/data: the sweep's systems are gb200_g{1,2,4,8},
# whose yamls live in this directory, and AIC resolves a system's data_dir
# relative to the root where it found that yaml. So this directory's `data` is
# what the sweep actually reads -- the overlay only takes effect through it.
#
# Usage (from anywhere; needs AFD_MEGAMOE_ROOT):
#   bash make_overlay.sh
#   sbatch add_moeperf_ep24.sbatch     # EP=24 rows interpolated from AIC's EP=16/32
#   sbatch add_ep24.sbatch             # EP=24 rows measured, DeepSeek-V4-Pro megamoe
set -euo pipefail
TOOLDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$TOOLDIR/../.." && pwd)
SHIPPED=$REPO/aic-core/src/aiconfigurator_core/systems/data
OV=${AFD_MEGAMOE_ROOT:?set AFD_MEGAMOE_ROOT to the data root}/aic_overlay

test -d "$SHIPPED" || { echo "no shipped systems data at $SHIPPED"; exit 1; }
if [ -e "$OV/data" ]; then
  echo "$OV/data already exists -- remove it first to rebuild (it may hold the EP=24 files)"
  exit 1
fi
mkdir -p "$OV"
cp -as "$SHIPPED" "$OV/data"
ln -sfn "$OV/data" "$TOOLDIR/data"

echo "overlay mirror: $OV/data  ($(find "$OV/data" -type l | wc -l) file links)"
echo "data link:      $TOOLDIR/data -> $(readlink "$TOOLDIR/data")"
echo "next: sbatch add_moeperf_ep24.sbatch && sbatch add_ep24.sbatch"
