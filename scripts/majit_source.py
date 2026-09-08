"""Identify Cargo's resolved majit source, including optional local patches."""

from functools import cache
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


@cache
def resolved_source() -> tuple[Path, str]:
    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    packages = json.loads(result.stdout)["packages"]
    package = next(p for p in packages if p["name"] == "majit-metainterp")
    directory = Path(package["manifest_path"]).parent
    if package["source"]:
        return directory, package["source"].rsplit("#", 1)[-1]
    head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain", "-uno"], text=True)
    return directory, head + ("-dirty" if dirty else "")
