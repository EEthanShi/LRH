#!/usr/bin/env bash
# Run a command with HOME pointing at a workspace in which the original server folder names
# resolve to this checkout, so the unmodified experiment code finds its files.
#
# Usage:  scripts/lrh_env.sh <command> [args...]
#         scripts/lrh_env.sh bash -c 'cd ~/hrh_v2_stage1_20261008 && venv/bin/python -m pytest -q hrhv2/tests'
#
# Workspace: $LRH_WORKSPACE (default <repo>/.lrh_home). Inside it:
#   hrh_v2_stage1_20261008 -> <repo>/stage1          hrh_v2_stage2_20261008 -> <repo>/stage2
#   hrh_v2_figs            -> <repo>/figures         hrh_v2_figs_venv       -> <repo>/figures/venv
#   hrh_finite_20260922/vendor -> <repo>/stage1/vendor   (PRH and ARH reference code, see scripts/fetch_vendor.sh)
#   .cache                 -> $LRH_REAL_HOME/.cache  (keeps the Hugging Face cache in the real home)
# No experiment file is edited. Existing entries in the workspace are left as they are.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${LRH_WORKSPACE:-$REPO/.lrh_home}"
REAL_HOME="${LRH_REAL_HOME:-$HOME}"
mkdir -p "$WS" "$WS/hrh_finite_20260922" "$REAL_HOME/.cache"
link() { [ -e "$WS/$1" ] || [ -L "$WS/$1" ] || ln -s "$2" "$WS/$1"; }
link hrh_v2_stage1_20261008     "$REPO/stage1"
link hrh_v2_stage2_20261008     "$REPO/stage2"
link hrh_v2_figs                "$REPO/figures"
link hrh_v2_figs_venv           "$REPO/figures/venv"
link hrh_finite_20260922/vendor "$REPO/stage1/vendor"
link .cache                     "$REAL_HOME/.cache"
export LRH_REAL_HOME="$REAL_HOME"
exec env HOME="$WS" "$@"
