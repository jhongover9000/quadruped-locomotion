"""Run logging: TensorBoard, a progress.csv for quick analysis, and a console line.

Every algorithm logs through this, so runs can be compared side by side:

    tensorboard --logdir outputs
"""

from __future__ import annotations

import csv
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: Path, csv_columns: list[str]):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(run_dir))
        self.csv_columns = csv_columns
        self._csv_file = open(Path(run_dir) / "progress.csv", "a", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_file, fieldnames=csv_columns, extrasaction="ignore")
        if self._csv_file.tell() == 0:
            self._csv.writeheader()

    def log(self, step: int, scalars: dict[str, float]) -> None:
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, step)
        row = {k.split("/")[-1]: v for k, v in scalars.items()}
        row["iteration"] = step
        self._csv.writerow(row)
        self._csv_file.flush()

    def close(self) -> None:
        self.writer.close()
        self._csv_file.close()
