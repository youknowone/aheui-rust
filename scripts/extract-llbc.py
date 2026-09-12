#!/usr/bin/env python3
"""Aheui crate/layout configuration for the shared llbc_extract engine.

Run from an Aheui checkout nested in Pyre. Artifacts go to build/llbc and
include the runtime helpers consumed by aheui-jit/build.rs; Charon installation and freshness checks
are owned by the shared engine in the parent checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The aheui workspace sits inside the pyre checkout, in CI and in a local
# worktree alike, so the engine and the Charon install resolve the same way in
# both.
PYRE_ROOT = ROOT.parent
sys.path.insert(0, str(PYRE_ROOT / "scripts"))

from llbc_extract import CrateSpec, run_cli  # noqa: E402


# CI redirects the pinned majit dependencies at its own checkout, and the
# extraction has to compile against the same crates the build will. The path is
# CI's to name, so it arrives in the environment rather than being spelled here.
_CARGO_CONFIG = os.environ.get("AHEUI_CARGO_CONFIG")
_CONFIG_ARGS = ["--config", _CARGO_CONFIG] if _CARGO_CONFIG else []

SPECS: dict[str, CrateSpec] = {
    # The graph consumer is aheui-jit, so the runtime is extracted with the
    # JIT-only hint markers (`dont_look_inside`, elidable and the related
    # policies) present; without the feature they are compiled out and the
    # translation loses them.
    "aheui-runtime": CrateSpec(
        name="aheui-runtime",
        crate_dir=ROOT / "aheui-runtime",
        output_name="aheui-runtime.ullbc",
        # Extraction compiles the runtime on its own. Select its host backend
        # here, without forcing dynasm into every consumer's feature graph.
        cargo_args=["--features", "jit,majit-metainterp/dynasm", *_CONFIG_ARGS],
    ),
    # Optional full-interpreter extraction for translator census work. The
    # helper artifact build does not consume this macro-expanded engine code.
    # Keep the JIT layout when explicitly requested.
    "aheuinterpreter": CrateSpec(
        name="aheuinterpreter",
        crate_dir=ROOT / "aheuinterpreter",
        output_name="aheuinterpreter.ullbc",
        cargo_args=["--features", "jit,majit-metainterp/dynasm", *_CONFIG_ARGS],
    ),
}

DEFAULT_CRATES = ["aheui-runtime"]

# Targets, besides the extraction host, that get a layout sidecar. `build/llbc`
# is read by every build of `aheui-jit`, including the wasm32 one, and a wasm32
# pointer is 4 bytes: `ListBase` is `{head: *mut Node, size: u32}`, so `size`
# sits at offset 4 there and at offset 8 on a 64-bit host. Without its own
# offsets the descr names the wrong word — past the end of the struct, in that
# case — and the JIT writes into whatever follows it.
LAYOUT_TARGETS = ("wasm32-wasip1",)

# Workspace resolution is a compiler input. `Cargo.lock` is not tracked here,
# so the manifest alone carries it.
BASE_PATHSPECS = ["Cargo.toml"]

# Bump only when aheui's extraction behaviour changes in a way the cargo and
# Charon flags the engine already hashes do not represent.
EXTRACTION_ABI = "1"


def main() -> None:
    run_cli(
        SPECS,
        DEFAULT_CRATES,
        root=ROOT,
        out_dir=ROOT / "build" / "llbc",
        extraction_abi=EXTRACTION_ABI,
        base_pathspecs=BASE_PATHSPECS,
        charon_root=PYRE_ROOT,
        layout_targets=LAYOUT_TARGETS,
    )


if __name__ == "__main__":
    main()
