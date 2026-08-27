#!/usr/bin/env python3
"""aheui driver for the Charon ULLBC extraction engine.

Declares the aheui crate table and delegates to the neutral engine in the
sibling pyre checkout's `scripts/llbc_extract.py`. Artefacts land under
`<aheui>/build/llbc` and are read by `aheui-jit/build.rs` through
`MAJIT_MIR_FRONTEND_LLBC`.

The engine lives in pyre rather than here because it is the half that is not
aheui's: platform keys, the Charon install layout, the nightly-skew crate
attribute, and the source fingerprint that decides whether an artefact is
still current. Only the crate table below is aheui's, which is the split
pyre's own driver documents ("external consumer repos carry their own
drivers"). aheui already reaches into that checkout for the Charon binary, so
this adds no path the extraction did not already depend on.
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
        cargo_args=["--features", "jit", *_CONFIG_ARGS],
    ),
    # The naive `mainloop` portal the pipeline traces.
    "aheuinterpreter": CrateSpec(
        name="aheuinterpreter",
        crate_dir=ROOT / "aheuinterpreter",
        output_name="aheuinterpreter.ullbc",
        cargo_args=list(_CONFIG_ARGS),
    ),
}

DEFAULT_CRATES = ["aheui-runtime", "aheuinterpreter"]

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
