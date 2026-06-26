from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


DEFAULT_VIS_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default_vis.yaml"


def load_vis_config(path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = _load_yaml(DEFAULT_VIS_CONFIG_PATH)
    if path is not None:
        config = _deep_merge(config, _load_yaml(Path(path)))
    if overrides:
        config = _deep_merge(config, overrides)
    _validate_config(config)
    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load visualization configs.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Visualization config {path} must contain a YAML mapping.")
    return data


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _validate_config(config: dict[str, Any]) -> None:
    if float(config.get("fps", 0)) <= 0:
        raise ValueError("fps must be positive.")
    viewer = str(config.get("viewer", "file")).lower()
    if viewer not in {"file", "usd", "viser", "gl", "null"}:
        raise ValueError("viewer must be one of file, usd, viser, gl, or null.")
    if int(config.get("port", 0)) <= 0:
        raise ValueError("port must be positive.")
