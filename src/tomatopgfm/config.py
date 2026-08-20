from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from .model import TomatoPGFMConfig


def _filter_known(data: dict, cls) -> dict:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("rt", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_model_config(path: str | Path) -> TomatoPGFMConfig:
    """Build a TomatoPGFMConfig from a configs/model.yaml file.

    The YAML carries presentation/run keys (model_name, striped_pattern,
    graph_modes, rc_policy, target_*) that are not model hyperparameters; those
    are ignored here. Only keys matching TomatoPGFMConfig fields are consumed, so the
    configs are now actually *read* (previously they were dead documentation).
    """
    raw = load_yaml(path)
    cfg = TomatoPGFMConfig(**_filter_known(raw, TomatoPGFMConfig))
    if cfg.graph_feature_dim <= 0:
        raise ValueError(f"graph_feature_dim must be > 0, got {cfg.graph_feature_dim}")
    return cfg
