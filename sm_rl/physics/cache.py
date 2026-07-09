"""Persistent, process-safe-ish cache for computed hadron spectra.

Keyed by the canonical model key (see spectrum.canonical_key); values are numpy
spectra. Since each miss costs a Mathematica evaluation, persisting this across
runs (and merging across data-parallel workers) is a large saving.

Usage:
    cache = SpectrumCache("runs/foo/cache.pkl")   # loads if present
    env = SMModelEnv(engine, cfg, cache=cache)
    ...
    cache.flush()                                  # also auto-flushes every N writes

Drop-in for a plain dict: supports `in`, `[]`, `[]=`. Auto-flushes every
`flush_every` new entries and merges with the on-disk file on flush, so multiple
workers pointed at the same path accumulate each other's results (last-writer
wins on the rare key collision, which is harmless -- the spectrum is
deterministic).
"""
from __future__ import annotations

import os
import pickle


class SpectrumCache(dict):
    def __init__(self, path: str | None = None, flush_every: int = 200,
                 merge_on_flush: bool = True):
        super().__init__()
        self.path = path
        self.flush_every = flush_every
        self.merge_on_flush = merge_on_flush
        self._writes_since_flush = 0
        self.n_hits = 0        # lookups that hit
        self.n_writes = 0      # distinct spectra computed & stored (i.e. misses)
        if path and os.path.exists(path):
            self.load()

    def load(self) -> None:
        with open(self.path, "rb") as f:
            super().update(pickle.load(f))

    def __contains__(self, key) -> bool:
        present = super().__contains__(key)
        if present:
            self.n_hits += 1
        return present

    def __setitem__(self, key, value) -> None:
        new = not super().__contains__(key)
        super().__setitem__(key, value)
        if new:
            self.n_writes += 1
            self._writes_since_flush += 1
            if self.path and self.flush_every and self._writes_since_flush >= self.flush_every:
                self.flush()

    def flush(self) -> None:
        if not self.path:
            return
        data = dict(self)
        if self.merge_on_flush and os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    disk = pickle.load(f)
                disk.update(data)          # our (fresh) entries win on collision
                data = disk
                super().update(disk)       # absorb other workers' entries into memory
            except Exception:
                pass                       # a corrupt/partial file must not kill training
        tmp = f"{self.path}.tmp.{os.getpid()}"
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)         # atomic
        self._writes_since_flush = 0

    def stats(self) -> dict:
        total = self.n_hits + self.n_writes
        return {
            "cache_size": len(self),
            "cache_hits": self.n_hits,
            "cache_miss": self.n_writes,
            "cache_hit_rate": round(self.n_hits / total, 4) if total else 0.0,
        }
