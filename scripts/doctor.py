"""Preflight check for running sm_rl (especially on a cluster).

Verifies: Python + packages, the Wolfram kernel, the GroupMath/spinSinglets
files, and a live engine evaluation. Prints an actionable PASS/FAIL summary.
Exits nonzero if any required check fails.

Run:  uv run python scripts/doctor.py        (or: python scripts/doctor.py)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import importlib.util
import platform
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, WARN, FAIL = "  OK ", " WARN", " FAIL"
_fail = 0


def line(status, msg):
    print(f"[{status}] {msg}")


def check(cond, msg, fatal=True):
    global _fail
    if cond:
        line(OK, msg)
    else:
        line(FAIL if fatal else WARN, msg)
        if fatal:
            _fail += 1
    return bool(cond)


def main():
    print("=" * 68)
    print("sm_rl doctor")
    print("=" * 68)
    print(f"platform : {platform.platform()}")
    print(f"python   : {sys.version.split()[0]}  ({sys.executable})")
    print(f"env      : WOLFRAM_KERNEL={os.environ.get('WOLFRAM_KERNEL','(unset)')}")
    print(f"           WOLFRAM_MATH_DIR={os.environ.get('WOLFRAM_MATH_DIR','(unset)')}")
    print("-" * 68)

    # packages
    for pkg in ("numpy", "scipy", "wolframclient"):
        check(importlib.util.find_spec(pkg) is not None,
              f"package '{pkg}' importable", fatal=True)
    has_torch = importlib.util.find_spec("torch") is not None
    check(has_torch, "package 'torch' importable (needed only for training: uv sync --extra rl)",
          fatal=False)

    # config-resolved paths
    from sm_rl.config import EngineConfig
    cfg = EngineConfig(verbose=False)
    print("-" * 68)
    print(f"kernel_path     : {cfg.kernel_path}")
    print(f"mathematica_dir : {cfg.mathematica_dir}")
    print("-" * 68)

    kernel_ok = check(os.path.exists(cfg.kernel_path) or shutil.which(cfg.kernel_path),
                      "Wolfram kernel found", fatal=True)
    files_ok = True
    for fn in (cfg.groupmath_file, cfg.spinsinglets_file):
        p = os.path.join(cfg.mathematica_dir, fn)
        files_ok &= check(os.path.exists(p), f"{fn} present in mathematica_dir", fatal=True)

    # live engine eval
    if kernel_ok and files_ok and importlib.util.find_spec("wolframclient"):
        print("-" * 68)
        try:
            from sm_rl.physics import WolframEngine
            eng = WolframEngine(cfg).start()
            try:
                reps = eng.reps_up_to_dim("SU3", 10)
                spins = eng.get_spins("SU3", [(1, 0), (0, 1)], [2, 2], [1, 2])
                ok = len(reps) >= 2 and set(dict(spins)) == {1, 3}
                check(ok, f"live engine eval (SU3 meson -> {spins})", fatal=True)
            finally:
                eng.stop()
        except Exception as e:
            check(False, f"live engine eval raised: {type(e).__name__}: {e}", fatal=True)

    print("=" * 68)
    if _fail == 0:
        print("DOCTOR: ALL REQUIRED CHECKS PASSED"
              + ("" if has_torch else "  (install torch for training: uv sync --extra rl)"))
        sys.exit(0)
    else:
        print(f"DOCTOR: {_fail} REQUIRED CHECK(S) FAILED — see [FAIL] lines above")
        sys.exit(1)


if __name__ == "__main__":
    main()
