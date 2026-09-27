import os
import pathlib
import yaml

_BASE_DIR = pathlib.Path(__file__).resolve().parent
_CONFIG_PATH = os.environ.get("CONFIG_PATH", str(_BASE_DIR / "config.yaml"))

def _load_raw():
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def _resolve(path_str):
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else (_BASE_DIR / p).resolve()

class Config:
    def __init__(self):
        raw = _load_raw()
        self.model_artifact_path = _resolve(
            os.environ.get("MODEL_ARTIFACT_PATH", raw["model"]["artifact_path"]))
        self.low_confidence_threshold = float(
            os.environ.get("LOW_CONFIDENCE_THRESHOLD", raw["model"]["low_confidence_threshold"]))
        self.candidate_tiles_dir = _resolve(
            os.environ.get("CANDIDATE_TILES_DIR", raw["training"]["candidate_tiles_dir"]))
        self.test_size = float(raw["training"]["test_size"])
        self.random_state = int(raw["training"]["random_state"])
        self.sqlite_path = _resolve(
            os.environ.get("SQLITE_PATH", raw["storage"]["sqlite_path"]))
        self.api_host = os.environ.get("API_HOST", raw["api"]["host"])
        self.api_port = int(os.environ.get("API_PORT", raw["api"]["port"]))

config = Config()