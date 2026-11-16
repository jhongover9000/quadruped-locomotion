"""Repository paths, so scripts work no matter where they are launched from.

Set ``QUADRUPED_OUTPUT_DIR`` to write training runs somewhere else (on a cluster, the
scratch file system: see hpc/README.md).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
ASSET_DIR = REPO_ROOT / "assets"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(os.environ.get("QUADRUPED_OUTPUT_DIR") or REPO_ROOT / "outputs").expanduser()
POLICY_DIR = REPO_ROOT / "policies"
