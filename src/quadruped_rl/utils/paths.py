"""Repository paths, so scripts work no matter where they are launched from."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
ASSET_DIR = REPO_ROOT / "assets"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
POLICY_DIR = REPO_ROOT / "policies"
