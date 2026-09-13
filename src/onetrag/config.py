"""Configuration loading: config.yaml + .env, with a stable hash for caching.

config_hash() covers every experimental parameter, so any change to
config.yaml invalidates cached LLM calls and distinguishes result files.
Paths (data_dir, repo root) are excluded from the hash: they vary by machine
but do not change what an experiment computes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    load_dotenv(REPO_ROOT / ".env")
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = os.getenv("ONET_DATA_DIR") or cfg["onet"].get("data_dir") or "../db_30_3_csv"
    p = Path(data_dir)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    cfg["onet"]["data_dir"] = str(p)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def config_hash(cfg: dict) -> str:
    hashable = {k: v for k, v in cfg.items() if not k.startswith("_")}
    hashable["onet"] = {k: v for k, v in hashable["onet"].items() if k != "data_dir"}
    blob = json.dumps(hashable, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def data_path(cfg: dict, *parts: str) -> Path:
    """Path under Research/data/, created on demand."""
    p = Path(cfg["_repo_root"]) / "data"
    for part in parts:
        p = p / part
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def results_path(cfg: dict, *parts: str) -> Path:
    p = Path(cfg["_repo_root"]) / "results"
    for part in parts:
        p = p / part
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
