#!/usr/bin/env bash
# Stage the 5 Bonn RGB-D Dynamic sequences for CUT3R Table 2 (video depth).
#
# CUT3R/MonST3R Table 2 evaluate Bonn on 5 sequences. The bonn loader reads
# $BONN_ROOT/rgbd_bonn_<name>/{rgb,depth,groundtruth.txt}.
#
# Source: per-sequence zips (public, no account) from ipb.uni-bonn.de
# (rgbd_dynamic2019). ~270 MB each, ~1.4 GB total — far smaller than the
# 16 GB all-sequences rgbd_bonn_dataset.zip.
#
# Usage:
#   BONN_ROOT=~/data/bonn_cut3r/root scripts/stage_bonn_cut3r.sh

set -euo pipefail

ROOT="${BONN_ROOT:-$HOME/data/bonn_cut3r/root}"
B="https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019"
SEQS=(balloon2 crowd2 crowd3 person_tracking2 synchronous)

mkdir -p "$ROOT/.dl"
for s in "${SEQS[@]}"; do
    z="$ROOT/.dl/rgbd_bonn_${s}.zip"
    [ -s "$z" ] || curl -sL "$B/rgbd_bonn_${s}.zip" -o "$z"
    unzip -n -q "$z" -d "$ROOT"   # -> rgbd_bonn_<name>/{rgb,depth,groundtruth.txt}
done

echo ">> done. Set BONN_ROOT=$ROOT and run: plumbline reproduce cut3r-bonn-110"
