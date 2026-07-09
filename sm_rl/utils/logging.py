"""File-backed run logging.

A Logger owns a run directory and writes:
  run_dir/
    log.txt          human-readable, timestamped (also echoed to stdout)
    metrics.csv      one row per logged step (scalars)
    config.json      run configuration
    best_model.json  best model found so far (updated on improvement)
    checkpoints/     PPO checkpoints

Everything is flushed immediately so a killed job leaves complete logs.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime


class Logger:
    def __init__(self, run_dir: str | None = None, name: str = "run", also_stdout: bool = True):
        if run_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join("runs", f"{name}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.run_dir = run_dir
        self.also_stdout = also_stdout
        self._log_path = os.path.join(run_dir, "log.txt")
        self._csv_path = os.path.join(run_dir, "metrics.csv")
        self._csv_fields: list[str] | None = None
        self._logf = open(self._log_path, "a")
        self.log(f"run dir: {os.path.abspath(run_dir)}")

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        if self.also_stdout:
            print(line, flush=True)
        self._logf.write(line + "\n")
        self._logf.flush()

    def log_metrics(self, row: dict) -> None:
        """Append a row of scalars to metrics.csv (header fixed on first call)."""
        if self._csv_fields is None:
            self._csv_fields = list(row.keys())
            need_header = not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0
            self._csvf = open(self._csv_path, "a", newline="")
            self._writer = csv.DictWriter(self._csvf, fieldnames=self._csv_fields)
            if need_header:
                self._writer.writeheader()
        self._writer.writerow({k: row.get(k) for k in self._csv_fields})
        self._csvf.flush()

    def save_json(self, name: str, obj) -> None:
        with open(os.path.join(self.run_dir, name), "w") as f:
            json.dump(obj, f, indent=2, default=str)

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.run_dir, "checkpoints")

    def close(self) -> None:
        self._logf.close()
        if getattr(self, "_csvf", None):
            self._csvf.close()
