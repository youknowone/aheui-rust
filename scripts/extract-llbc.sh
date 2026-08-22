#!/usr/bin/env bash
# extract-llbc.sh — run Charon on the aheui crates the JIT graph pipeline
# consumes and drop `.ullbc` artefacts into ./build/llbc/.
#
# The majit-translate graph pipeline (aheui-jit/build.rs) lowers
# Charon-extracted MIR (`.ullbc`) into a SemanticProgram. This script is
# the producer of those `.ullbc` files; the consumer is the build script
# via `PYRE_MIR_FRONTEND_LLBC`.
#
# Usage:
#   scripts/extract-llbc.sh                   # extract both aheui crates
#   scripts/extract-llbc.sh aheui-runtime     # extract one crate
#   LLBC_DEST=./out scripts/extract-llbc.sh   # override output dir
#
# Notes:
#   - Charon invokes `cargo build` internally under its pinned nightly
#     toolchain (installed by the parent repo's
#     `python3 scripts/install-charon.py`).
#   - Outputs are NOT committed (see /build/ in .gitignore). Re-run after
#     source changes in aheuinterpreter / aheui-runtime; Cargo's
#     incremental cache keeps re-runs cheap.

set -euo pipefail

aheui_root="$(cd "$(dirname "$0")/.." && pwd)"
# The aheui workspace lives inside the parent majit repo; Charon is
# installed there by scripts/install-charon.py.
parent_root="$(cd "$aheui_root/.." && pwd)"

case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) platform_key="darwin-arm64" ;;
    Linux-x86_64) platform_key="linux-x86_64" ;;
    Linux-aarch64|Linux-arm64) platform_key="linux-aarch64" ;;
    *)
        echo "extract-llbc.sh: unsupported host $(uname -s)-$(uname -m)" >&2
        exit 1
        ;;
esac
shared_build="${PYRE_SHARED_BUILD:-$(cd "$parent_root/.." && pwd)/.pyre-build}"
charon_dest="${CHARON_DEST:-$shared_build/charon/$platform_key}"
charon_bin="${CHARON_BIN:-$charon_dest/charon}"
if [[ ! -x "$charon_bin" ]]; then
    echo "extract-llbc.sh: charon not installed at $charon_bin" >&2
    echo "  run the parent repo's: python3 scripts/install-charon.py" >&2
    echo "  or set CHARON_BIN to a charon binary." >&2
    exit 1
fi

LLBC_DEST="${LLBC_DEST:-$aheui_root/build/llbc}"
case "$LLBC_DEST" in
    /*) ;;
    *) LLBC_DEST="$aheui_root/$LLBC_DEST" ;;
esac
mkdir -p "$LLBC_DEST"

# Crate name -> path. The build pipeline reads aheuinterpreter (the naive
# `mainloop` portal) + aheui-runtime (linked-list storage / Node ops).
crate_path() {
    case "$1" in
        aheui-runtime)   echo "$aheui_root/aheui-runtime" ;;
        aheuinterpreter) echo "$aheui_root/aheuinterpreter" ;;
        *)               echo "" ;;
    esac
}

ALL_CRATES="aheui-runtime aheuinterpreter"
if [[ "$#" -eq 0 ]]; then
    targets="$ALL_CRATES"
else
    targets="$*"
fi

# Match the parent extract-llbc.sh `cfg_select!` skew workaround: inject
# `#![feature(cfg_select)]` into every crate compiled during extraction so
# the pinned-nightly build does not trip E0658 on transitive deps. No-op
# for crates that don't use the macro; affects ONLY the charon build.
export RUSTC_BOOTSTRAP=1
charon_crate_attr="-Zcrate-attr=feature(cfg_select)"
if [[ -n "${RUSTFLAGS:-}" ]]; then
    export RUSTFLAGS="$RUSTFLAGS $charon_crate_attr"
else
    export RUSTFLAGS="$charon_crate_attr"
fi
export CARGO_UNSTABLE_HOST_CONFIG=true
export CARGO_UNSTABLE_TARGET_APPLIES_TO_HOST=true
charon_host_config=(
    --config target-applies-to-host=false
    --config "host.rustflags=[\"$charon_crate_attr\"]"
)

for crate in $targets; do
    path="$(crate_path "$crate")"
    if [[ -z "$path" ]]; then
        echo "extract-llbc.sh: unknown crate '$crate'" >&2
        echo "  known: $ALL_CRATES" >&2
        exit 1
    fi
    if [[ ! -d "$path" ]]; then
        echo "extract-llbc.sh: missing crate dir for '$crate' at $path" >&2
        exit 1
    fi

    dest="$LLBC_DEST/${crate}.ullbc"
    echo "=== extracting $crate -> $dest ==="

    pushd "$path" > /dev/null
    cargo_features=()
    if [[ "$crate" == "aheui-runtime" ]]; then
        # The graph consumer is aheui-jit, so retain the JIT-only hint
        # markers (`dont_look_inside`, elidable, and related policies) in the
        # runtime snapshot it translates.
        cargo_features=(--features jit)
    fi
    cargo_config=()
    if [[ -n "${AHEUI_CARGO_CONFIG:-}" ]]; then
        cargo_config=(--config "$AHEUI_CARGO_CONFIG")
    fi
    "$charon_bin" cargo --ullbc --dest-file "$dest" -- \
        "${cargo_features[@]}" "${cargo_config[@]}" "${charon_host_config[@]}"
    popd > /dev/null

    size="$(du -h "$dest" | cut -f1)"
    echo "    wrote $dest ($size)"
done

echo
echo "all extractions complete. artefacts under: $LLBC_DEST"
