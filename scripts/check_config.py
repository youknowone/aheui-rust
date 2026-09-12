"""Validate dependency pins shared by Cargo, extraction CI and WASM packaging."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path = ROOT) -> None:
    manifest = (root / "Cargo.toml").read_text()
    # These declarations are intentionally single-line inline tables. Reject
    # other shapes rather than silently overlooking an unchecked dependency.
    entries = re.findall(r"^majit-[\w-]+\s*=.*$", manifest, re.MULTILINE)
    assert entries, "no majit workspace dependencies"
    revisions = set()
    for entry in entries:
        match = re.search(r'\brev\s*=\s*"([0-9a-f]{40})"', entry)
        assert match, f"majit needs an exact Git SHA: {entry}"
        revisions.add(match[1])
    assert len(revisions) == 1, f"majit revisions disagree: {revisions}"
    revision = next(iter(revisions))
    assert revision and re.fullmatch(r"[0-9a-f]{40}", revision), "majit needs an exact Git SHA"
    workflow = (root / ".github/workflows/snippet-matrix.yml").read_text()
    assert f"ref: {revision}\n" in workflow, "CI checkout differs from Cargo's majit pin"
    wasm = (root / "aheui-wasm/Cargo.toml").read_text()
    match = re.search(r'^wasm-bindgen\s*=\s*"([^"]+)"', wasm, re.MULTILINE)
    assert match, "missing wasm-bindgen dependency pin"
    version = match[1]
    assert version.startswith("="), "wasm-bindgen schema requires an exact pin"
    for name in (".github/workflows/pages.yml", "aheui-wasm/build.sh"):
        text = (root / name).read_text()
        assert f"--version {version[1:]}" in text, f"{name}: wasm-bindgen CLI disagrees"


if __name__ == "__main__":
    check()
    print("dependency and packaging pins agree")
