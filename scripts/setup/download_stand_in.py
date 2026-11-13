"""Download the stand-in robot model from MuJoCo Menagerie (not committed to the repo).

    python scripts/setup/download_stand_in.py                 # Unitree Go1
    python scripts/setup/download_stand_in.py --robot unitree_go2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from quadruped_rl.utils.paths import ASSET_DIR

MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie.git"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", default="unitree_go1", help="folder name in MuJoCo Menagerie")
    p.add_argument("--force", action="store_true", help="re-download if it already exists")
    args = p.parse_args()

    dest = ASSET_DIR / "robots" / "stand_in" / args.robot
    if dest.exists() and not args.force:
        print(f"Already present: {dest} (use --force to re-download)")
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        MENAGERIE, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", args.robot], check=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(Path(tmp) / args.robot, dest)
    print(f"Downloaded {args.robot} to {dest}")


if __name__ == "__main__":
    main()
