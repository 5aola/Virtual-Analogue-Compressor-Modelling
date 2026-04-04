"""
B_sota_analysis utilities — re-exports shared functions from ``src.*`` and
defines analysis-specific constants.

All imports that previously used ``from B_sota_analysis.utils import ...`` or
``import utils as u`` continue to work.
"""

# Analysis-specific constants
SSL_DSET_PATH = "/Volumes/Production Tools/coding_projs/THESIS/data_preprocesses/data/Diff-SSL-G-Comp"
SR = 44100
window_size = 1024
overlap = 0.75

# Re-export everything from src modules so existing callers keep working
from src.audio_io import collect_audio_files, get_audio_stats, load_audio  # noqa: F401
from src.dsp import (  # noqa: F401
    calc_gain_reduction,
    calculate_rms,
    estimate_attack_release_time,
    moving_rms,
    rms_to_db,
    to_amplitude,
    to_dB,
    window_peak,
    window_rms,
)
from src.plotting import (  # noqa: F401
    compare_gain_curves,
    estimate_gain_curve,
    regression,
)
from src.transfer import FRAC, H1, auto_fft, time_varying_transfer  # noqa: F401
