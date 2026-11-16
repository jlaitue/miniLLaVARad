from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_paths(path=None):
    path = Path(path) if path else REPO_ROOT / "configs" / "paths.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Copy configs/paths.example.yaml and edit it.")
    return OmegaConf.load(path)
