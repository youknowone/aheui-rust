"""Shared text format for Aheui benchmark baselines."""

from pathlib import Path


def parse_fields(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def format_fields(fields: dict) -> str:
    return "".join(f"{key}={value}\n" for key, value in sorted(fields.items()))


def read_int_fields(path: Path) -> dict[str, int]:
    return {key: int(value) for key, value in parse_fields(path.read_text()).items()}


def write_fields(path: Path, fields: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_fields(fields))
