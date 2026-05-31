# Source before GPU sessions on k8s pods:
#   source /mnt/localssd/plumbline/scripts/pod-localssd-env.sh
#
# Home overlay (~75G) fills fast; stage datasets, runs, clones, and HF/uv
# caches under /mnt/localssd/plumbline-work/. Stay within 1/8 of the node's
# local SSD (~3 TiB on a 26 TiB volume) or the pod may be killed.

export PLUMBLINE_WORK="${PLUMBLINE_WORK:-/mnt/localssd/plumbline-work}"

mkdir -p \
  "$PLUMBLINE_WORK"/{data,runs,deps,cache/huggingface,cache/uv,predictions}

# Caches (avoid refilling $HOME/.cache after a quota wipe)
export HF_HOME="$PLUMBLINE_WORK/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$PLUMBLINE_WORK/cache/torch"
export UV_CACHE_DIR="$PLUMBLINE_WORK/cache/uv"
export XDG_CACHE_HOME="$PLUMBLINE_WORK/cache/xdg"
export HF_HUB_DISABLE_XET=1

# HF token: paste once into $PLUMBLINE_WORK/.hf_token (chmod 600). Never commit.
if [[ -r "${PLUMBLINE_WORK}/.hf_token" ]]; then
  export HF_TOKEN="$(<"${PLUMBLINE_WORK}/.hf_token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# MoGe-eval bundles (ROOT = parent of DDAD/, Sintel/, …)
export DDAD_MOGE_ROOT="$PLUMBLINE_WORK/data/moge_eval"
export SINTEL_MOGE_ROOT="$PLUMBLINE_WORK/data/moge_eval"
# ETH3D MoGe bundle lives under data/eth3d_moge/ (not moge_eval/).
export ETH3D_MOGE_ROOT="$PLUMBLINE_WORK/data/eth3d_moge"
# DIODE MoGe bundle is mirrored at datasets/diode_moge/ (not under moge_eval/).
export DIODE_MOGE_ROOT="${DIODE_MOGE_ROOT:-$PLUMBLINE_WORK/data/diode_moge}"
export KITTI_MOGE_ROOT="$PLUMBLINE_WORK/data/moge_eval"
export IBIMS1_ROOT="${IBIMS1_ROOT:-$PLUMBLINE_WORK/data/ibims1}"
export BOOSTER_ROOT="${BOOSTER_ROOT:-$PLUMBLINE_WORK/data/booster}"
export MIDDLEBURY_ROOT="${MIDDLEBURY_ROOT:-$PLUMBLINE_WORK/data/middlebury}"
export ETH3D_ROOT="${ETH3D_ROOT:-$PLUMBLINE_WORK/data/eth3d}"

export DAV2_ROOT="${DAV2_ROOT:-$PLUMBLINE_WORK/deps/depth-anything-v2}"
export SINTEL_ROOT="${SINTEL_ROOT:-$PLUMBLINE_WORK/data/sintel}"
