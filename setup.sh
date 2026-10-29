#!/usr/bin/env bash
# One-step setup: checks the prerequisites, builds, makes the tooling venv (build/tools/venv), exports the robot
# meshes for the viewer and, when the BONES-SEED shapes are present, bakes the SOMA body surface the retargeter
# needs. Safe to re-run: finished steps are skipped (--force redoes the generated assets).
set -euo pipefail
cd "$(dirname "$0")"

build=1 surface=1 force=0
for arg in "$@"; do
    case "$arg" in
        --no-build) build=0 ;;
        --no-surface) surface=0 ;;
        --force) force=1 ;;
        -h|--help)
            cat <<'EOF'
usage: ./setup.sh [--no-build] [--no-surface] [--force]
  --no-build    skip the xmake build
  --no-surface  skip the SOMA body surface (installs torch and SOMA-X, ~3 GB)
  --force       regenerate the robot meshes and the body surface
Environment: SOMA_BVH_ROOT  SOMA uniform BVH clips (default ~/Downloads/soma_uniform/bvh)
EOF
            exit 0 ;;
        *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
    esac
done

step() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
bvh_root=${SOMA_BVH_ROOT:-$HOME/Downloads/soma_uniform/bvh}
venv=build/tools/venv

step "Prerequisites"
missing=()
need() { command -v "$1" >/dev/null 2>&1 || missing+=("$1: $2"); }
need swiftc "Swift 6 toolchain (https://www.swift.org/install, e.g. swiftly)"
need xmake "xmake 3 (https://xmake.io)"
need cmake "cmake, used to build slang-rhi"
need git "git"
need python3 "Python 3.10 or newer"
if ((${#missing[@]})); then
    printf '    missing %s\n' "${missing[@]}" >&2
    exit 1
fi
swift_major=$(swiftc --version 2>&1 | sed -n 's/.*Swift version \([0-9]*\).*/\1/p' | head -1)
if [[ -z "$swift_major" || "$swift_major" -lt 6 ]]; then
    echo "    Swift 6 or newer is needed (found: $(swiftc --version 2>&1 | head -1))" >&2
    exit 1
fi
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "    Python 3.10 or newer is needed" >&2; exit 1; }
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    note "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    note "no NVIDIA driver found: retargeting and the viewer's contact analysis need a CUDA GPU"
fi
note "Swift $(swiftc --version 2>&1 | sed -n 's/.*Swift version \([^ ]*\).*/\1/p' | head -1), $(xmake --version | head -1 | sed 's/\x1b\[[0-9;]*m//g' | cut -d, -f1), $(python3 --version)"

if ((build)); then
    step "Build (first run downloads and builds Slang and slang-rhi)"
    xmake f -m release -y >/dev/null
    xmake -y
fi

step "Tooling venv ($venv)"
[[ -x $venv/bin/python ]] || python3 -m venv "$venv"
"$venv/bin/pip" install -q --upgrade pip
"$venv/bin/pip" install -q -r tools/requirements.txt
"$venv/bin/pip" install -q -e python
note "ready: numpy, usd-core, pyzmq, retargeter_client"

step "Robot meshes for the viewer"
for robot in g1_29dof g1_23dof; do
    out=build/viewer/robots/$robot.glb
    if [[ -f $out && $force == 0 ]]; then note "$out exists"; continue; fi
    mkdir -p build/viewer/robots
    "$venv/bin/python" -I tools/export_robot_meshes.py "assets/robots/unitree_g1/${robot/dof/_dof}" "$out"
done

step "SOMA body surface"
shapes=assets/body_models/soma/seed_soma_shapes/soma_base_fit_mhr_params.npz
surface_file=assets/body_models/soma/surface_lo.npz
clip=$bvh_root/230322/sweep_floor_side_walk_R_004__A298.bvh
if ((!surface)); then
    note "skipped (--no-surface)"
elif [[ -f $surface_file && $force == 0 ]]; then
    note "$surface_file exists"
elif [[ ! -f $shapes ]]; then
    note "skipped: the BONES-SEED shapes are not in assets/body_models/soma/seed_soma_shapes/ (licensed, not in git)."
    note "Retargeting, the viewer's body and most tests need $surface_file; copy the shapes there and re-run."
elif [[ ! -f $clip ]]; then
    note "skipped: $clip not found (set SOMA_BVH_ROOT to the SOMA uniform BVH folder)"
else
    "$venv/bin/pip" install -q -r tools/requirements-soma.txt
    [[ -d build/tools/SOMA-X ]] || git clone --depth 1 https://github.com/NVlabs/SOMA-X.git build/tools/SOMA-X
    "$venv/bin/pip" install -q --no-deps -e build/tools/SOMA-X
    mkdir -p build/fixtures
    "$venv/bin/python" -I tools/bake_soma_surface.py --soma-x build/tools/SOMA-X/assets --shape "$shapes" \
        --bvh "$clip" --out "$surface_file" --fixture build/fixtures/soma_lbs_reference.npz
fi

step "Done"
note "xmake test                                      run the test suites"
note "build/linux/x86_64/release/retarget view         viewer on http://127.0.0.1:8080"
note "build/linux/x86_64/release/retarget --help       all commands"
