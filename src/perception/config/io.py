# src/perception/config/io.py
import json
from pathlib import Path
from .model import AppConfig

def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        return AppConfig()
    data = json.loads(p.read_text(encoding="utf-8"))
    return AppConfig(**data)

def save_config(cfg: AppConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Pydantic v2: dump to a dict, then json.dumps so we can use ensure_ascii=False
    txt = json.dumps(cfg.model_dump(), ensure_ascii=False, indent=2)
    p.write_text(txt, encoding="utf-8")
