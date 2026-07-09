"""sm_rl: RL search over gauge groups + matter content to rediscover the
Standard Model quark sector from the gauge-singlet hadron spectrum."""

import os as _os

# conda macOS commonly links libomp via both numpy(MKL) and scipy/torch; this
# benign setdefault avoids the "OMP: Error #15" abort. Set before heavy imports.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .config import Config, EngineConfig, EnvConfig, GroupConfig, RewardConfig

__all__ = ["Config", "EngineConfig", "EnvConfig", "GroupConfig", "RewardConfig"]
