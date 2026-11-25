"""Physics engine: a thin, cached Python wrapper around Mathematica/GroupMath.

The engine exposes exactly the group-theory primitives the RL framework needs
and nothing more, so the rest of the codebase never imports wolframclient and
can be exercised against `MockEngine` in unit tests.

All engine results are deterministic functions of their arguments, so every
method is memoized. `get_spins` is the expensive one (a GroupMath
`PermutationSymmetryOfInvariants` call); caching it is essential, not optional.
"""
from __future__ import annotations

import abc
import os
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

from ..config import EngineConfig

Rep = tuple[int, ...]          # a rep as GroupMath Dynkin labels, e.g. (1, 0) for SU(3) fund


def to_mathematica(obj) -> str:
    """Convert Python scalars / lists / tuples / arrays to Wolfram list syntax."""
    if isinstance(obj, (list, tuple, np.ndarray)):
        return "{" + ", ".join(to_mathematica(x) for x in obj) + "}"
    if isinstance(obj, (Fraction,)):
        return f"({obj.numerator}/{obj.denominator})"
    if isinstance(obj, float) and obj.is_integer():
        return str(int(obj))
    return str(obj)


def _to_number(x):
    """Coerce a wolframclient scalar (int / float / Fraction-like) to a Python number."""
    if isinstance(x, Fraction):
        return x
    try:
        # wolframclient returns rationals as a special expr; fall back to float.
        return int(x) if float(x).is_integer() else float(x)
    except (TypeError, ValueError):
        return x


def _to_int_tuple(x) -> tuple[int, ...]:
    return tuple(int(v) for v in x)


class Engine(abc.ABC):
    """Abstract group-theory oracle."""

    @abc.abstractmethod
    def start(self) -> "Engine": ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def reps_up_to_dim(self, group: str, max_dim: int) -> tuple[Rep, ...]:
        """Dynkin labels of all irreps with dim <= max_dim, sorted by dimension.
        Index 0 is the trivial rep, index 1 the fundamental, and so on."""

    @abc.abstractmethod
    def dim(self, group: str, rep: Rep) -> int: ...

    @abc.abstractmethod
    def dynkin_index(self, group: str, rep: Rep): ...

    @abc.abstractmethod
    def casimir(self, group: str, rep: Rep): ...

    @abc.abstractmethod
    def conjugate_rep(self, group: str, rep: Rep) -> Rep: ...

    @abc.abstractmethod
    def conjugacy_class(self, group: str, rep: Rep) -> tuple[int, ...]:
        """Center charge / N-ality vector of the rep (governs singlet formation)."""

    @abc.abstractmethod
    def adjoint(self, group: str) -> Rep: ...

    @abc.abstractmethod
    def get_spins(
        self,
        group: str,
        reps: Sequence[Rep],
        spins: Sequence[int],
        flavours: Sequence[int],
    ) -> tuple[tuple[int, int], ...]:
        """Gauge-singlet combinations respecting spin-statistics for the given
        multiset of partons. Returns ((su2_spin_label, multiplicity), ...) where
        the SU(2) label is 2S+1 (spin 0 -> 1, 1/2 -> 2, 1 -> 3, 3/2 -> 4)."""

    # ---- derived helpers (engine-agnostic) --------------------------------

    def min_copies_for_singlet(self, group: str, rep: Rep, cutoff: int = 50) -> int:
        """Fewest copies of `rep` whose tensor product contains a singlet.
        SU(3)->3, SO(N)->2, E6->3, ... Used to set the U(1) charge quantum."""
        raise NotImplementedError

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def canonical_spins_args(group: str, reps, spins, flavours):
    """Canonical form of a GetSpins query. GetSpins is a function of the
    *multiset* of partons, and the flavour ids only encode which entries are the
    same field, so the result is invariant under (i) permuting the parton list
    and (ii) relabelling flavours. Sort the (rep, spin, flavour) triples and
    relabel flavours by first appearance. Returns (key, reps, spins, flavours)
    with the lists in canonical order, so a memo keyed on `key` is hit by every
    equivalent query (e.g. the same operator built from partons in a different
    model order), which also keeps the persisted memo small.
    Verified against the real engine by scripts/validate_spins_memo.py."""
    triples = sorted(zip((tuple(int(x) for x in r) for r in reps),
                         (int(s) for s in spins), (int(f) for f in flavours)))
    relabel: dict[int, int] = {}
    creps, cspins, cflavs = [], [], []
    for r, sp, f in triples:
        if f not in relabel:
            relabel[f] = len(relabel) + 1
        creps.append(r); cspins.append(sp); cflavs.append(relabel[f])
    key = ("spins", group, tuple(creps), tuple(cspins), tuple(cflavs))
    return key, creps, cspins, cflavs


class WolframEngine(Engine):
    """Real GroupMath-backed engine."""

    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self._session = None
        self._cache: dict[tuple, object] = {}
        self.n_spins_calls = 0   # real GetSpins evaluations this process
        self.n_spins_hits = 0    # GetSpins queries answered from the memo

    # ---- lifecycle --------------------------------------------------------

    def _preflight(self) -> None:
        """Fail early with actionable guidance if the kernel / GroupMath are missing."""
        import shutil

        k = self.cfg.kernel_path
        if not (os.path.exists(k) or shutil.which(k)):
            raise FileNotFoundError(
                f"Wolfram kernel not found at {k!r}. Put 'WolframKernel' on PATH, "
                f"set $WOLFRAM_KERNEL, or pass EngineConfig(kernel_path=...)."
            )
        for fn in (self.cfg.groupmath_file, self.cfg.spinsinglets_file):
            p = os.path.join(self.cfg.mathematica_dir, fn)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{fn} not found in {self.cfg.mathematica_dir!r}. Set $WOLFRAM_MATH_DIR "
                    f"or pass EngineConfig(mathematica_dir=...) to the GroupMath folder."
                )

    def start(self) -> "WolframEngine":
        if self._session is not None:
            return self
        self._preflight()
        import time

        from wolframclient.evaluation import WolframLanguageSession
        from wolframclient.language import wl, wlexpr

        self._wlexpr = wlexpr
        attempts = max(1, self.cfg.start_retries)
        last_err = None
        for attempt in range(1, attempts + 1):
            if self.cfg.verbose:
                print(f"[engine] starting Wolfram kernel (attempt {attempt}/{attempts}, "
                      f"STARTUP_TIMEOUT={self.cfg.startup_timeout}s): {self.cfg.kernel_path}")
            try:
                # STARTUP_TIMEOUT is forwarded to the kernel controller as a parameter.
                self._session = WolframLanguageSession(
                    kernel=self.cfg.kernel_path,
                    STARTUP_TIMEOUT=self.cfg.startup_timeout,
                )
                # The first evaluate is what actually forces the kernel to launch,
                # so kernel-startup failures surface here, not at construction.
                self._session.evaluate(wl.SetDirectory(self.cfg.mathematica_dir))
                self._session.evaluate(wl.Get(self.cfg.groupmath_file))
                # Load GetSpins DownValues exactly as the GT-Invariants notebook does.
                self._session.evaluate(
                    wlexpr(
                        "Module[{rules}, rules = Get[\"%s\"]; DownValues[GetSpins] = rules;]"
                        % self.cfg.spinsinglets_file
                    )
                )
                if self.cfg.verbose:
                    print("[engine] GroupMath + spinSinglets loaded.")
                return self
            except Exception as e:  # kernel launch/socket timeout -> tear down + retry
                last_err = e
                print(f"[engine] kernel start failed on attempt {attempt}/{attempts}: {e}")
                self._teardown_session()
                if attempt < attempts:
                    delay = self.cfg.start_retry_backoff * attempt
                    print(f"[engine] retrying in {delay:.0f}s ...")
                    time.sleep(delay)
        raise RuntimeError(
            f"Wolfram kernel failed to start after {attempts} attempts: {last_err}"
        )

    def _teardown_session(self) -> None:
        """Best-effort terminate of a half-started session before a retry."""
        if self._session is not None:
            try:
                self._session.terminate()
            except Exception:
                pass
            self._session = None

    def stop(self) -> None:
        if self._session is not None:
            self._session.terminate()
            self._session = None

    def _eval(self, expr: str):
        if self._session is None:
            self.start()
        return self._session.evaluate(self._wlexpr(expr))

    # ---- primitives (all memoized) ---------------------------------------

    def reps_up_to_dim(self, group: str, max_dim: int) -> tuple[Rep, ...]:
        key = ("reps", group, max_dim)
        if key not in self._cache:
            raw = self._eval(f"RepsUpToDimN[{group}, {max_dim}]")
            reps = tuple(_to_int_tuple(r) for r in raw)
            # RepsUpToDimN returns dimension-sorted with trivial first; keep order.
            self._cache[key] = reps
        return self._cache[key]

    def dim(self, group: str, rep: Rep) -> int:
        key = ("dim", group, rep)
        if key not in self._cache:
            self._cache[key] = int(self._eval(f"DimR[{group}, {to_mathematica(rep)}]"))
        return self._cache[key]

    def dynkin_index(self, group: str, rep: Rep) -> float:
        key = ("dynkin", group, rep)
        if key not in self._cache:
            # N[...] so wolframclient returns a plain float, not a Rational expr.
            self._cache[key] = float(self._eval(f"N[DynkinIndex[{group}, {to_mathematica(rep)}]]"))
        return self._cache[key]

    def casimir(self, group: str, rep: Rep) -> float:
        key = ("casimir", group, rep)
        if key not in self._cache:
            self._cache[key] = float(self._eval(f"N[Casimir[{group}, {to_mathematica(rep)}]]"))
        return self._cache[key]

    def conjugate_rep(self, group: str, rep: Rep) -> Rep:
        key = ("conj", group, rep)
        if key not in self._cache:
            self._cache[key] = _to_int_tuple(
                self._eval(f"ConjugateIrrep[{group}, {to_mathematica(rep)}]")
            )
        return self._cache[key]

    def conjugacy_class(self, group: str, rep: Rep) -> tuple[int, ...]:
        key = ("cclass", group, rep)
        if key not in self._cache:
            raw = self._eval(f"ConjugacyClass[{group}, {to_mathematica(rep)}]")
            self._cache[key] = _to_int_tuple(raw) if isinstance(raw, (list, tuple)) else (int(raw),)
        return self._cache[key]

    def adjoint(self, group: str) -> Rep:
        key = ("adjoint", group)
        if key not in self._cache:
            self._cache[key] = _to_int_tuple(self._eval(f"Adjoint[{group}]"))
        return self._cache[key]

    def get_spins(self, group, reps, spins, flavours) -> tuple[tuple[int, int], ...]:
        key, creps, cspins, cflavs = canonical_spins_args(group, reps, spins, flavours)
        hit = self._cache.get(key)
        if hit is not None:
            self.n_spins_hits += 1
            return hit
        rep_str = ", ".join(to_mathematica(r) for r in creps)
        spin_str = ", ".join(to_mathematica(int(s)) for s in cspins)
        flav_str = ", ".join(to_mathematica(int(f)) for f in cflavs)
        raw = self._eval(
            f"GetSpins[{group}, {{{rep_str}}}, {{{spin_str}}}, {{{flav_str}}}]"
        )
        out = tuple((int(lbl), int(mult)) for lbl, mult in raw) if raw else ()
        self._cache[key] = out
        self.n_spins_calls += 1
        return out

    def get_spins_raw(self, group, reps, spins, flavours) -> tuple[tuple[int, int], ...]:
        """Un-memoized, un-canonicalized GetSpins call (validation only)."""
        rep_str = ", ".join(to_mathematica(r) for r in reps)
        spin_str = ", ".join(to_mathematica(int(s)) for s in spins)
        flav_str = ", ".join(to_mathematica(int(f)) for f in flavours)
        raw = self._eval(f"GetSpins[{group}, {{{rep_str}}}, {{{spin_str}}}, {{{flav_str}}}]")
        return tuple((int(lbl), int(mult)) for lbl, mult in raw) if raw else ()

    # ---- memo persistence ---------------------------------------------------
    # Everything in self._cache is a pure function of (group, reps, spins) --
    # charge-free -- so it is tiny (thousands of entries, KB-MB) and can be
    # shared across runs. Persisting it is what makes the big charge-keyed
    # spectrum cache unnecessary: a spectrum "miss" then costs only Python.

    def memo_stats(self) -> dict:
        n_spins = sum(1 for k in self._cache if k[0] == "spins")
        return {"memo_size": len(self._cache), "memo_spins": n_spins,
                "spins_calls": self.n_spins_calls, "spins_hits": self.n_spins_hits}

    def load_memo(self, path: str) -> int:
        """Merge a persisted memo into memory. Returns entries loaded (0 if absent)."""
        import pickle
        if not path or not os.path.exists(path):
            return 0
        with open(path, "rb") as f:
            disk = pickle.load(f)
        self._cache.update(disk)
        return len(disk)

    def save_memo(self, path: str) -> int:
        """Atomically write the memo, merged with whatever is on disk (so several
        workers can share one file). Returns the number of entries written."""
        import pickle
        if not path:
            return 0
        data = dict(self._cache)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    disk = pickle.load(f)
                disk.update(data)
                data = disk
                self._cache.update(disk)
            except Exception:
                pass
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return len(data)

    def min_copies_for_singlet(self, group: str, rep: Rep, cutoff: int = 50) -> int:
        key = ("mincopies", group, rep)
        if key in self._cache:
            return self._cache[key]
        for n in range(1, cutoff + 1):
            rep_str = ", ".join(to_mathematica(rep) for _ in range(n))
            res = self._eval(f"PermutationSymmetryOfInvariants[{group}, {{{rep_str}}}]")
            # res == {labels, invariants}; nonempty invariants => a singlet exists.
            try:
                has = len(res[1]) > 0
            except (TypeError, IndexError):
                has = False
            if has:
                self._cache[key] = n
                return n
        raise ValueError(f"No singlet within {cutoff} copies of {rep} in {group}")
