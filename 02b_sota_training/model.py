"""Build the diffssl LSTM32TVC / LSTM96TVC model from the published nablafx package.

The local ``external/nablafx-for-diffssl-compressor`` checkout is gitignored;
Colab installs ``nablafx`` from PyPI (same architecture as the diffssl
``experiments/LSTM{32,96}TVC`` configs).
"""

from __future__ import annotations

from nablafx.core.models import BlackBoxModel
from nablafx.processors import LSTM


def build_diffssl_tvc_lstm(
    hidden_size: int = 32,
    num_layers: int = 1,
    num_controls: int = 4,
    cond_block_size: int = 128,
    cond_num_layers: int = 1,
) -> BlackBoxModel:
    """``LSTM32TVC`` / ``LSTM96TVC`` recipe: tvcond + 4 static knobs."""
    processor = LSTM(
        num_inputs=1,
        num_outputs=1,
        num_controls=num_controls,
        hidden_size=hidden_size,
        num_layers=num_layers,
        residual=False,
        direct_path=False,
        cond_type="tvcond",
        cond_block_size=cond_block_size,
        cond_num_layers=cond_num_layers,
    )
    return BlackBoxModel(processor=processor)
