"""mamba_va: a selective state-space model for nonlinear, time-variant audio
effects (analog dynamic-range compressors), improving on Mamba / S6.

See README.md and DESIGN.md for the rationale and the link to the
nonlinear state-space realization theory (Shoukry, 2008).
"""

from .model import CompSSM
from .blocks import CompSSMBlock
from .ssm import SelectiveSSM
from .detector import AdaptiveLevelDetector, level_db_norm
from .film import FiLM

__all__ = [
    "CompSSM",
    "CompSSMBlock",
    "SelectiveSSM",
    "AdaptiveLevelDetector",
    "level_db_norm",
    "FiLM",
]
__version__ = "0.2.0"
