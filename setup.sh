#!/usr/bin/env bash
# One-step setup on Linux (CUDA) or macOS (Metal): fetches the tools it can (xmake, uv and through it Python, the
# Metal Toolchain), builds, makes the tooling venv (build/tools/venv), exports the robot meshes for the viewer,
# downloads the BONES-SEED shapes (gated on Hugging Face) and bakes the SOMA body surface the retargeter needs.
# Downloaded tools are pinned, checksummed and kept in build/tools/bin; nothing outside the repository is changed
# except the Metal Toolchain (an Xcode component). Safe to re-run: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"
root=$PWD

build=1 surface=1 force=0 motions=
default_bvh_root=$HOME/Downloads/soma_uniform/bvh
while (($#)); do
    case "$1" in
        --no-build) build=0 ;;
        --no-surface) surface=0 ;;
        --force) force=1 ;;
        --download-motions)
            motions=${default_bvh_root%/bvh}
            if [[ $# -gt 1 && $2 != --* ]]; then motions=$2; shift; fi ;;
        -h|--help)
            cat <<'EOF'
usage: ./setup.sh [--no-build] [--no-surface] [--force] [--download-motions [DIR]]
  --no-build             skip the xmake build
  --no-surface           skip the SOMA body surface (baked from the BONES-SEED shapes)
  --force                regenerate the robot meshes and the body surface
  --download-motions DIR download and unpack the BONES-SEED SOMA uniform clips (45 GB archive) into DIR
                         (default ~/Downloads/soma_uniform; the clips end up in DIR/bvh)
Environment: SOMA_BVH_ROOT  SOMA uniform BVH clips (default ~/Downloads/soma_uniform/bvh)
             HF_TOKEN       Hugging Face token for the gated BONES-SEED dataset (default: `hf auth login`)
EOF
            exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

step() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() { printf '    %s\n' "$@" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

tools=build/tools
tool_bin=$root/$tools/bin
venv=$tools/venv
mkdir -p "$tool_bin"
export PATH="$tool_bin:$PATH"
bvh_root=${SOMA_BVH_ROOT:-${motions:+$motions/bvh}}
bvh_root=${bvh_root:-$default_bvh_root}

os=$(uname -s)
arch=$(uname -m)
[[ $arch == aarch64 ]] && arch=arm64
case "$os/$arch" in
    Linux/x86_64|Linux/arm64|Darwin/arm64|Darwin/x86_64) ;;
    *) die "unsupported platform $os/$arch (Linux x86_64 / arm64 with CUDA, macOS with Metal)" ;;
esac

sha256() { if have sha256sum; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
# fetch URL SHA256 FILE: download and verify (the file only appears once it matches).
fetch() {
    curl -fsSL --retry 3 -o "$3.part" "$1" || die "download failed: $1"
    local got
    got=$(sha256 "$3.part")
    [[ $got == "$2" ]] || { rm -f "$3.part"; die "checksum mismatch for $1" "expected $2, got $got"; }
    mv "$3.part" "$3"
}

step "Platform"
if [[ $os == Darwin ]]; then
    xcode-select -p >/dev/null 2>&1 || die "the Xcode command line tools are missing: xcode-select --install"
    note "GPU: $(system_profiler SPDisplaysDataType 2>/dev/null | sed -n 's/.*Chipset Model: //p' | head -1) (Metal)"
    if ! xcrun -sdk macosx metal --version >/dev/null 2>&1; then
        note "installing the Metal Toolchain (Xcode component, ~700 MB): slang-rhi compiles kernels with it"
        xcodebuild -downloadComponent MetalToolchain >/dev/null ||
            die "could not install it; needs Xcode (not only the command line tools): https://developer.apple.com/xcode/"
    fi
elif have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    note "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (CUDA)"
else
    note "no NVIDIA driver found: retargeting and the viewer's contact analysis need a CUDA GPU (CPU backend only)"
fi

step "Toolchain"
have git || die "git is missing"
if ! have swiftc; then
    if [[ $os == Darwin ]]; then die "Swift is missing: install Xcode (https://developer.apple.com/xcode/)"
    else die "Swift is missing: install swiftly (https://www.swift.org/install/linux/), then \`swiftly install latest\`"; fi
fi
swift_version=$(swiftc --version 2>&1 | sed -n 's/.*Swift version \([0-9][0-9.]*\).*/\1/p' | head -1)
[[ ${swift_version%%.*} -ge 6 ]] 2>/dev/null || die "Swift 6 or newer is needed (found: ${swift_version:-none})"

# xmake 3 (cmake, used for slang-rhi, comes through xmake's own packages when it is missing). Leftover XMAKE_* variables
# of an older installation point a newer xmake at the old scripts; the build ignores them.
xmake_major() { env -u XMAKE_PROGRAM_DIR -u XMAKE_ROOTDIR -u XMAKE_EXE "$1" --version 2>/dev/null |
    sed -n 's/.*xmake v\([0-9]*\)\..*/\1/p' | head -1; }
for var in XMAKE_PROGRAM_DIR XMAKE_ROOTDIR; do
    [[ -n ${!var:-} && ! -d ${!var} ]] && note "warning: $var=${!var} does not exist (left over from an old xmake?)"
done
unset XMAKE_PROGRAM_DIR XMAKE_ROOTDIR XMAKE_EXE
xmake_version=3.1.2
xmake=
if have xmake && [[ $(xmake_major xmake) -ge 3 ]] 2>/dev/null; then
    xmake=$(command -v xmake)
else
    case "$os/$arch" in
        Linux/x86_64) bundle=linux.x86_64 sum=ac216aa71f4954cc1129aabf1099571d618211c1ef8db8fba871377ec5fb47cd ;;
        Darwin/arm64) bundle=macos.arm64 sum=e719121c56cd34f5f0c0731954a154c1df1b0915f303e5c82dfb6c6cb9e3ef9e ;;
        Darwin/x86_64) bundle=macos.x86_64 sum=ba2e282a7e706d99a17f39fb92e432c9815628407ab7e9e09d9bc3b0270f9339 ;;
        *) die "xmake 3 is needed and there is no prebuilt one for $os/$arch: https://xmake.io/guide/quick-start" ;;
    esac
    if [[ ! -x $tool_bin/xmake || $(xmake_major "$tool_bin/xmake") != 3 ]]; then
        note "fetching xmake $xmake_version into $tools/bin"
        fetch "https://github.com/xmake-io/xmake/releases/download/v$xmake_version/xmake-bundle-v$xmake_version.$bundle" \
            "$sum" "$tool_bin/xmake"
        chmod +x "$tool_bin/xmake"
    fi
    xmake=$tool_bin/xmake
fi

# uv: makes the venv, with a Python of its own when the system has none new enough (the pins need 3.12).
uv_version=0.13.4
if have uv; then
    uv=$(command -v uv)
else
    case "$os/$arch" in
        Linux/x86_64) triple=x86_64-unknown-linux-gnu sum=2b428396f12572d4429cf210ca234dd599a33b3af0958a4d5ed569d63e3f344f ;;
        Linux/arm64) triple=aarch64-unknown-linux-gnu sum=1ae72b4718729e2ccc4f1ae2c6d1e531ca8467800e0546e9130b3886023a60ab ;;
        Darwin/arm64) triple=aarch64-apple-darwin sum=e0b1d4743ce64c55d9f0f4a444afa6b81125680070ee04e853fcd0f6c1ebf8f1 ;;
        Darwin/x86_64) triple=x86_64-apple-darwin sum=7be2dcd316122f4b3e5a0a7889f6248266c14425374e105ff7dc9779184603a6 ;;
    esac
    if [[ ! -x $tool_bin/uv ]]; then
        note "fetching uv $uv_version into $tools/bin"
        fetch "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-$triple.tar.gz" "$sum" "$tools/uv.tar.gz"
        tar -xzf "$tools/uv.tar.gz" -C "$tool_bin" --strip-components 1 "uv-$triple/uv"
        rm "$tools/uv.tar.gz"
    fi
    uv=$tool_bin/uv
fi
note "Swift $swift_version, xmake $("$xmake" --version | head -1 | sed 's/\x1b\[[0-9;]*m//g; s/.*xmake v\([^ ,]*\).*/\1/'), uv $("$uv" --version | cut -d' ' -f2)"

if ((build)); then
    step "Build (first run downloads and builds Slang and slang-rhi)"
    "$xmake" f -m release -y
    "$xmake" -y
fi

step "Tooling venv ($venv)"
if [[ ! -x $venv/bin/python ]] || ! "$venv/bin/python" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
    rm -rf "$venv"
    "$uv" venv -q --python 3.12 "$venv"
fi
# Synced to exactly the lock file (requirements.txt resolved for every platform): also drops what older setups put
# there (torch, SOMA-X: ~1 GB).
"$uv" pip sync -q --python "$venv/bin/python" tools/requirements.lock
"$uv" pip install -q --python "$venv/bin/python" -e python
note "ready: Python $("$venv/bin/python" -c 'import platform; print(platform.python_version())'), numpy, usd-core, pyzmq, huggingface_hub, retargeter_client"

step "Robot meshes for the viewer"
for robot in g1_29dof g1_23dof; do
    out=build/viewer/robots/$robot.glb
    if [[ -f $out && $force == 0 ]]; then note "$out exists"; continue; fi
    mkdir -p build/viewer/robots
    "$venv/bin/python" -I tools/export_robot_meshes.py "assets/robots/unitree_g1/${robot/dof/_dof}" "$out"
done

# BONES-SEED (gated: accept the licence on the dataset page, then `hf auth login` or HF_TOKEN).
seed_repo=bones-studio/seed
seed_page=https://huggingface.co/datasets/$seed_repo
hf() { "$venv/bin/hf" "$@"; }
seed_hint() { note "$1" "accept the licence at $seed_page, log in (build/tools/venv/bin/hf auth login, or set HF_TOKEN)," \
    "then re-run ./setup.sh"; }

if [[ -n $motions && ! -d $motions/bvh ]]; then
    step "SOMA uniform clips ($motions/bvh)"
    free_gb=$(df -Pk "$(dirname "$motions")" 2>/dev/null | awk 'NR == 2 { print int($4 / 1048576) }')
    if [[ ${free_gb:-0} -lt 100 ]]; then
        note "skipped: ${free_gb:-?} GB free next to $motions; the 45 GB archive and its contents need ~100 GB"
    elif hf download "$seed_repo" soma_uniform.tar.gz --repo-type dataset --local-dir "$motions" >/dev/null; then
        note "unpacking ..."
        tar -xzf "$motions/soma_uniform.tar.gz" -C "$motions" && rm "$motions/soma_uniform.tar.gz"
        [[ -d $motions/bvh ]] || { [[ -d $motions/soma_uniform/bvh ]] && mv "$motions/soma_uniform/bvh" "$motions/bvh"; }
        [[ -d $motions/bvh ]] && note "clips in $motions/bvh" || note "unpacked into $motions, but found no bvh/ folder"
    else
        seed_hint "download failed:"
    fi
fi

step "SOMA body surface"
seed_dir=assets/body_models/soma/seed_soma_shapes
shapes=$seed_dir/soma_base_fit_mhr_params.npz
surface_file=assets/body_models/soma/surface_lo.npz
clip=$bvh_root/230322/sweep_floor_side_walk_R_004__A298.bvh

rm -rf "$tools/SOMA-X" "$tools/bake"   # SOMA-X and torch of older setups (up to 2.6 GB), no longer needed
if ((!surface)); then
    note "skipped (--no-surface)"
elif [[ -f $surface_file && $force == 0 ]]; then
    note "$surface_file exists"
else
    if [[ ! -f $shapes ]]; then
        note "fetching the BONES-SEED shapes (soma_shapes/, a few MB) from $seed_repo"
        rm -rf "$tools/seed"
        if hf download "$seed_repo" --repo-type dataset --include 'soma_shapes/*' --local-dir "$tools/seed" >/dev/null 2>&1 &&
            [[ -f $tools/seed/soma_shapes/soma_base_fit_mhr_params.npz ]]; then
            rm -rf "$seed_dir"
            mkdir -p "$(dirname "$seed_dir")"
            mv "$tools/seed/soma_shapes" "$seed_dir"
            rm -rf "$tools/seed"
        fi
    fi
    if [[ ! -f $shapes ]]; then
        seed_hint "skipped: no BONES-SEED shapes in $seed_dir/ (licensed, not in git; retargeting and most tests need them):"
    else
        # Pose source of the alignment check and the test fixture: a real clip when there is one (varied poses), else
        # the 2-frame base rig that comes with the shapes. The surface itself does not depend on it.
        if [[ -f $clip ]]; then bvh=$clip frames=0,100,400
        else bvh=$seed_dir/soma_base_rig/soma_base_skel_minimal.bvh frames=0,1; fi
        # numpy only: SOMA-X's body model comes as fixed arrays (soma_x_model.npz, tools/export_soma_x_model.py).
        mkdir -p build/fixtures
        "$venv/bin/python" -I tools/bake_soma_surface.py --shape "$shapes" --bvh "$bvh" --frames "$frames" \
            --out "$surface_file" --fixture build/fixtures/soma_lbs_reference.npz
    fi
fi

step "Done"
if [[ $xmake == "$tool_bin/xmake" ]]; then
    note "xmake was fetched into $tools/bin; to use it (and uv) directly: export PATH=\"$tool_bin:\$PATH\""
fi
note "xmake test                         run the test suites"
note "build/release/retarget view        viewer on http://127.0.0.1:8080"
note "build/release/retarget --help      all commands"
[[ -d $bvh_root ]] || note "the retargeting tests and examples read SOMA uniform clips from $bvh_root (SOMA_BVH_ROOT;" \
    "./setup.sh --download-motions fetches them)"
