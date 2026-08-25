#!/usr/bin/env python3
"""Run the pinned snippet corpus through aheui.aheui, the Aheui interpreter
written in Aheui.

What this covers that `snippet_matrix.py` does not: every snippet it runs is
executed twice over, once as the program under test and once as data for an
interpreter that is itself under test. A backend that is merely self-consistent
passes the direct corpus and fails here, because the metacircular run drives
storage selection, reflection and arithmetic through paths a hand-written
snippet reaches only in combination.

    scripts/metacircular.py                        # the whole corpus
    scripts/metacircular.py 99bottles standard     # a prefix filter
    scripts/metacircular.py --timeout 300

aheui.aheui is a separate project and is not vendored here. Point at a checkout
with `--aheui-aheui` or `AHEUI_AHEUI`; the default is a sibling directory:

    git clone https://github.com/aheui/aheui.aheui ../aheui.aheui

Two comparisons are reported, because they answer different questions:

  vs direct   byte-exact against the same binary running the snippet itself.
              This is the metacircular equivalence claim, and it is the one
              that fails when a backend is wrong.
  vs .out     against the committed expectation, with trailing newlines
              stripped from both sides. aheui.aheui's own test.sh compares
              through `$(...)`, which strips them, and several committed .out
              files carry a trailing newline that no implementation emits — so
              a byte-exact comparison here would report failures that are only
              a property of the corpus files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SNIPPETS = ROOT / "snippets"
BINARY = ROOT / "target" / "release" / "aheui"
DEFAULT_AHEUI_AHEUI = ROOT.parent / "aheui.aheui" / "aheui.aheui"

# Snippets deliberately left out, with the reason each one is out. A skip that
# does not say why is indistinguishable from a snippet nobody got round to
# fixing, and these two groups have opposite implications for this repository:
# the first says nothing about the backend, the second says the harness is
# correct and the interpreter it runs on is not.
SKIP = {
    # Metacircular execution costs a few hundred times the direct run, and
    # these two are the corpus's heaviest. `logo` did not finish in two hours
    # and `pi.jinseo` was killed by the system at 111 minutes. Both run
    # correctly when executed directly; there is nothing here to learn that
    # the direct corpus does not already report.
    "logo/logo": "runtime beyond a practical budget",
    "pi/pi.jinseo": "runtime beyond a practical budget",
    # aheui.aheui does not implement the no-op instructions other than `ㅇ`.
    # `ㄱ`, `ㄲ`, `ㅉ` and `ㅋ` are all 없음 in Aheui, but aheui.aheui treats
    # them as an operation needing an element and reflects the direction when
    # the selected storage is empty, so a program that passes through one of
    # those cells on an empty storage never reaches its 끝냄.
    #
    # Confirmed by rewriting only those initials to `ㅇ` — semantics-preserving,
    # since all five are the same no-op — which makes both snippets produce
    # their expected output under aheui.aheui. Reproduces identically under
    # rpaheui, so it is aheui.aheui's behaviour and not this backend's:
    #
    #     아아희  -> exits, 아가희 -> hangs, 바가희 -> exits
    "literature/ddeok": "aheui.aheui does not implement ㄱ/ㄲ/ㅉ/ㅋ as no-ops",
    "literature/sijo-div": "aheui.aheui does not implement ㄱ/ㄲ/ㅉ/ㅋ as no-ops",
}


def snippets(only: list[str]) -> list[tuple[str, Path]]:
    found = []
    for source in sorted(SNIPPETS.glob("*/*.aheui")):
        name = f"{source.parent.name}/{source.stem}"
        if only and not any(name.startswith(prefix) for prefix in only):
            continue
        if not source.with_suffix(".out").exists():
            continue
        found.append((name, source))
    return found


def run(argv: list[str], stdin: bytes, timeout: float):
    try:
        return subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("only", nargs="*", help="name prefixes to run")
    parser.add_argument(
        "--aheui-aheui",
        type=Path,
        default=Path(os.environ.get("AHEUI_AHEUI", DEFAULT_AHEUI_AHEUI)),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"no binary at {BINARY}; cargo build -p aheui --release")
        return 1
    if not args.aheui_aheui.exists():
        print(
            f"aheui.aheui not found at {args.aheui_aheui}\n"
            "  git clone https://github.com/aheui/aheui.aheui ../aheui.aheui\n"
            "  or pass --aheui-aheui / set AHEUI_AHEUI"
        )
        return 1

    ok = failed = 0
    for name, source in snippets(args.only):
        if name in SKIP:
            print(f"{name:<36} skip      {SKIP[name]}")
            continue

        stdin_path = source.with_suffix(".in")
        stdin = stdin_path.read_bytes() if stdin_path.exists() else b""
        # test.sh feeds the interpreter the program's source, a NUL, then the
        # program's own input. Without the separator aheui.aheui reads the
        # input as more source.
        payload = source.read_bytes() + (b"\0" + stdin if stdin_path.exists() else b"")

        direct = run([str(BINARY), str(source)], stdin, args.timeout)
        start = time.monotonic()
        meta = run([str(BINARY), str(args.aheui_aheui)], payload, args.timeout)
        elapsed = time.monotonic() - start

        if meta is None or direct is None:
            which = "aheui.aheui" if meta is None else "direct"
            print(f"{name:<36} TIMEOUT   {elapsed:7.2f}s  {which} exceeded {args.timeout:.0f}s")
            failed += 1
            continue

        expected = source.with_suffix(".out").read_bytes()
        detail = ""
        if meta.stdout != direct.stdout or meta.returncode != direct.returncode:
            detail += (
                f"vs direct: {len(meta.stdout)}B/exit{meta.returncode} != "
                f"{len(direct.stdout)}B/exit{direct.returncode}  "
            )
        if meta.stdout.rstrip(b"\n") != expected.rstrip(b"\n"):
            detail += f"vs .out: {len(meta.stdout)}B != {len(expected)}B"
        if detail:
            print(f"{name:<36} MISMATCH  {elapsed:7.2f}s  {detail}")
            failed += 1
        else:
            print(f"{name:<36} ok        {elapsed:7.2f}s")
            ok += 1

    print(f"\n{ok}/{ok + failed} agree with direct execution and the committed output")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
