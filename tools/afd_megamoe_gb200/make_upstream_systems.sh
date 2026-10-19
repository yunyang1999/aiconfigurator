#!/usr/bin/env bash
# Systems directory for running this sweep on another AIC checkout (upstream
# main, for Kimi-K3): the granularity files next to this script plus
#
#   gb200_nvl72.yaml   gb200 with the 72-GPU NVLink domain as one AIC node.
#                      Upstream refuses EP > num_gpus_per_node without DeepEP
#                      all-to-all tables for the model, and K3 has none. The
#                      colocated arm replaces dispatch, expert compute and
#                      combine with the measured stage, so only the gate moves;
#                      NVL72's NVLink bandwidth is the same within and across
#                      trays (900 GB/s in gb200.yaml either way).
#   data, attention_lane_defaults.yaml
#                      links into that checkout's packaged systems tree --
#                      upstream resolves data_dir against the directory the
#                      system yaml was found in.
#
# usage: AFD_MEGAMOE_ROOT=<bundle> AFD_AIC_ROOT=<aic checkout> make_upstream_systems.sh
set -euo pipefail
: "${AFD_MEGAMOE_ROOT:?}" "${AFD_AIC_ROOT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="${AFD_AIC_ROOT}/aic-core/src/aiconfigurator_core/systems"
OUT="${AFD_MEGAMOE_ROOT}/aic_systems_upstream"
[[ -d ${SRC}/data/gb200 ]] || { echo "no gb200 data under ${SRC}"; exit 2; }
mkdir -p "${OUT}"
cp "${HERE}"/gb200_g{1,2,4,8}.yaml "${OUT}/"
sed -e 's/^\(  num_gpus_per_node:\) 4$/\1 72/' \
    -e 's/OVERRIDE: 4 GPU(s) per AIC-node\./OVERRIDE: the 72-GPU NVLink domain as one AIC node./' \
    "${HERE}/gb200_g4.yaml" > "${OUT}/gb200_nvl72.yaml"
grep -q '^  num_gpus_per_node: 72$' "${OUT}/gb200_nvl72.yaml"
ln -sfn "${SRC}/data" "${OUT}/data"
ln -sfn "${SRC}/attention_lane_defaults.yaml" "${OUT}/attention_lane_defaults.yaml"
echo "systems dir ${OUT}:"
ls -l "${OUT}"
