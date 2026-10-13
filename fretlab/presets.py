"""Preset registry: discover, load, and scaffold instrument configs.

Presets live in ``fretlab/presets/*.yaml`` (packaged) and optionally in a user
directory (``~/.config/fretlab/presets``) for shop-specific instruments.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

PKG_PRESETS = Path(__file__).resolve().parent / "presets"


def user_preset_dir() -> Path:
    p = Path.home() / ".config" / "fretlab" / "presets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_presets() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in (PKG_PRESETS, user_preset_dir()):
        if d.is_dir():
            for f in sorted(d.glob("*.yaml")):
                out[f.stem] = f
    return out


def preset_info(name: str) -> dict:
    presets = list_presets()
    if name not in presets:
        raise KeyError(f"unknown preset '{name}'. available: {', '.join(presets)}")
    data = yaml.safe_load(presets[name].read_text(encoding="utf-8"))
    meta = data.get("preset", {})
    return {
        "name": name,
        "path": presets[name],
        "title": meta.get("title", name),
        "description": meta.get("description", ""),
    }


def scaffold(name: str, dest: str | Path, force: bool = False) -> Path:
    """Copy a preset to ``dest`` as a starting config."""
    info = preset_info(name)
    dest = Path(dest)
    if dest.is_dir():
        dest = dest / f"{name}.yaml"
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} exists (use --force)")
    shutil.copyfile(info["path"], dest)
    return dest
