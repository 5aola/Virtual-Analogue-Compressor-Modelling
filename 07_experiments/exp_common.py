"""Shared infrastructure for the 07_experiments inference-only notebooks.

Everything here runs LOCALLY on existing checkpoints — no training anywhere.
Importing this module bootstraps sys.path for the sibling numbered folders
(mirroring the 05/06 eval notebooks) and re-exports their public APIs:

- ``06_output``: ``GainPriorDiffSSLLSTM`` / ``GainPriorWSDiffSSLLSTM``,
  ``amplitude_match``, ``splits`` (byte-identical to 05's)
- ``05_conditioning``: ``DetectorGRLSTM`` (appended to sys.path, so its
  colliding module names — model/splits/dataset/system — stay shadowed by
  06_output's; only its unique modules resolve from there)
- ``03_initial_GR_pred``: ``eval_helpers`` (run listing, checkpoint picking,
  audio segment IO)

plus the canonical 9-column metric engine ported from
``02b_sota_training/eval_lstm_diffssl_tvc.ipynb`` (same code path as the 05/06
eval notebooks, verified against nablafx.evaluation there) and probe-signal /
analysis helpers used across the experiment notebooks.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

# ── sys.path bootstrap (mirrors 06_output/eval_lstm_gain_prior.ipynb cell 0) ──

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent

# Stub the broken nablafx import-chain wheels (rational / frechet_audio_distance)
_rational = types.ModuleType("rational")
_rational.torch = types.ModuleType("rational.torch")
_rational.torch.Rational = type("Rational", (), {})
sys.modules.setdefault("rational", _rational)
sys.modules.setdefault("rational.torch", _rational.torch)
_fad = types.ModuleType("frechet_audio_distance")
_fad.FrechetAudioDistance = type("FrechetAudioDistance", (), {})
sys.modules.setdefault("frechet_audio_distance", _fad)

for _p in (
    str(REPO_ROOT),                            # `src`
    str(REPO_ROOT / "nablafx"),                # raw ./nablafx clone (TVFiLMCond)
    str(REPO_ROOT / "06_output"),              # gain-prior models + splits
    str(REPO_ROOT / "09_end2end"),             # GainPriorWSE2ELSTM (predicted-GR cascade)
    str(REPO_ROOT / "03_initial_GR_pred"),     # eval_helpers
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# 05_conditioning is APPENDED: its unique modules (model_detector_gr, gr_target)
# resolve from there while its colliding names (model, splits, dataset, system)
# keep resolving to 06_output — same trick as eval_lstm_gain_prior cell 9a.
_p05 = str(REPO_ROOT / "05_conditioning")
if _p05 not in sys.path:
    sys.path.append(_p05)

# ── re-exports from the numbered folders ─────────────────────────────

from amplitude_match import (  # noqa: E402
    GR_DB_MAX, GR_DB_MIN, amplitude_match, gr_db_to_gain,
)
from model_gainprior import GainPriorDiffSSLLSTM            # noqa: E402
from model_gainprior_ws import GainPriorWSDiffSSLLSTM       # noqa: E402
from model_gainprior_ws_e2e import GainPriorWSE2ELSTM       # noqa: E402  (09_end2end)
from model_tfilm import GRTFiLMDiffSSLLSTM                  # noqa: E402  (06_output, GR temporal FiLM)
from model_detector_gr import DetectorGRLSTM                # noqa: E402
from model_blackbox_gr import BlackboxGRLSTM                # noqa: E402  (05_conditioning, appended path)
from splits import (  # noqa: E402
    DIFFSSL_PARAM_RANGES, SplitManifest, normalize_setting_params,
)
from eval_helpers import (  # noqa: E402
    _find_hparams_json, _pair_num_frames, _read_dry_wet_segment,
    _weighted_average_metric_rows, latest_best_checkpoint, list_runs,
    wet_dir_for_setting,
)
from src.dsp import PARAM_ORDER                             # noqa: E402
from src.dsp_torch import RMS_WINDOW, gain_reduction_db     # noqa: E402

# ── defaults ─────────────────────────────────────────────────────────

THESIS_DATA = Path("/Volumes/Saola's Drive/AllCode/thesis/data")
DATA_ROOT = str(THESIS_DATA / "Diff-SSL-G-Comp")
GAIN_PRIOR_RUNS = str(THESIS_DATA / "diffssl_gain_prior_runs")
E2E_RUNS = str(THESIS_DATA / "diffssl_e2e_runs")
GR_TFILM_RUNS = str(THESIS_DATA / "diffssl_gr_tfilm_runs")
GR_PRED_RUNS = str(THESIS_DATA / "gr_pred_runs")
SR = 44100
DEVICE = "cpu"
EVAL_OUT = EXP_DIR / "eval_output"

# Documented reference runs (override per notebook if needed)
DEFAULT_GAIN_PRIOR_RUN = "gain_prior_20260702_085618_diffssl_lstm32_gain_prior"
DEFAULT_DETECTOR_RUN = "lstm_gr_20260702_174039_lstm_detector_gr"
# 09_end2end predicted-GR cascade: frozen BlackboxGRLSTM (stage 1) -> GainPriorWSE2ELSTM (stage 2)
DEFAULT_E2E_RUN = "e2e_predgr_20260705_185738_diffssl_lstm32_ws_predgr"
DEFAULT_BLACKBOX_GR_RUN = "lstm_gr_20260705_125142_lstm_blackbox_gr_film"
DEFAULT_GR_TFILM_RUN = "gr_tfilm_20260701_195700_diffssl_lstm32_tvc_gr_tfilm"


def ensure_eval_out() -> Path:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    return EVAL_OUT


# ── checkpoint loading ───────────────────────────────────────────────


def _resolve_run(runs_dir: str, run_name: str | None, model_types: tuple[str, ...]) -> str:
    """Return the run dir path — `run_name` if given, else newest matching type."""
    if run_name:
        run_dir = os.path.join(runs_dir, run_name)
        assert os.path.isdir(run_dir), f"Run not found: {run_dir}"
        return run_dir
    df = list_runs(runs_dir)
    assert not df.empty, f"No runs in {runs_dir}"
    df = df[df["type"].isin(model_types)]
    assert not df.empty, f"No {model_types} runs in {runs_dir}"
    return str(df.iloc[-1]["path"])            # list_runs sorts by mtime asc


def load_gain_prior(run_name: str | None = DEFAULT_GAIN_PRIOR_RUN,
                    runs_dir: str = GAIN_PRIOR_RUNS, device: str = DEVICE):
    """Load a gain-prior checkpoint (additive or waveshaper variant).

    run_name=None picks the newest run of either type. Returns
    (model, hparams, run_dir); model in eval mode on `device`.
    """
    run_dir = _resolve_run(runs_dir, run_name,
                           ("GainPriorDiffSSLLSTM", "GainPriorWSDiffSSLLSTM"))
    with open(os.path.join(run_dir, "hparams.json")) as f:
        hp = json.load(f)
    m = hp["model"]
    common = dict(
        num_controls=int(m["num_controls"]),
        hidden_size=int(m["hidden_size"]),
        num_layers=int(m["num_layers"]),
        tvcond_dim=int(m.get("tvcond_dim", 16)),
        cond_block_size=int(m.get("cond_block_size", 128)),
        cond_num_layers=int(m.get("cond_num_layers", 1)),
        delta_max_db=float(m.get("delta_max_db", 12.0)),
        use_color=bool(m.get("use_color", True)),
    )
    if hp.get("model_type") == "GainPriorWSDiffSSLLSTM":
        model = GainPriorWSDiffSSLLSTM(
            **common, ws_hidden=int(m.get("ws_hidden", 8)),
            ws_film=bool(m.get("ws_film", False)),
        )
    else:
        assert hp.get("model_type") == "GainPriorDiffSSLLSTM", hp.get("model_type")
        model = GainPriorDiffSSLLSTM(**common)

    ckpt = latest_best_checkpoint(run_dir)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in sd.items()
                           if k.startswith("model.")})
    model.to(device).eval()
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,}-param "
          f"{hp['model_type']}  ({Path(run_dir).name} / {Path(ckpt).name})")
    return model, hp, run_dir


def load_gain_prior_e2e(run_name: str | None = DEFAULT_E2E_RUN,
                        runs_dir: str = E2E_RUNS, device: str = DEVICE):
    """Load a predicted-GR-cascade gain-prior model (``GainPriorWSE2ELSTM``, 09_end2end).

    Stage 2 of the deployed cascade. Same waveshaper core as
    ``load_gain_prior``'s ws variant, plus the ``use_matched_input`` flag; it was
    trained on the *frozen blackbox predictor's* GR rather than the oracle curve.
    ``hp['gr_frontend']`` records the exact stage-1 run + checkpoint — pass those
    to ``load_blackbox_gr`` to reconstruct the same cascade the run was trained as.
    Returns (model, hparams, run_dir); model in eval mode on ``device``.
    """
    run_dir = _resolve_run(runs_dir, run_name, ("GainPriorWSE2ELSTM",))
    with open(os.path.join(run_dir, "hparams.json")) as f:
        hp = json.load(f)
    assert hp.get("model_type") == "GainPriorWSE2ELSTM", hp.get("model_type")
    m = hp["model"]
    model = GainPriorWSE2ELSTM(
        use_matched_input=bool(m.get("use_matched_input", True)),
        num_controls=int(m["num_controls"]),
        hidden_size=int(m["hidden_size"]),
        num_layers=int(m["num_layers"]),
        tvcond_dim=int(m.get("tvcond_dim", 16)),
        cond_block_size=int(m.get("cond_block_size", 128)),
        cond_num_layers=int(m.get("cond_num_layers", 1)),
        delta_max_db=float(m.get("delta_max_db", 12.0)),
        use_color=bool(m.get("use_color", True)),
        ws_hidden=int(m.get("ws_hidden", 8)),
        ws_film=bool(m.get("ws_film", False)),
    )
    ckpt = latest_best_checkpoint(run_dir)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in sd.items()
                           if k.startswith("model.")})
    model.to(device).eval()
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,}-param "
          f"{hp['model_type']}  ({Path(run_dir).name} / {Path(ckpt).name})  "
          f"matched_input={bool(m.get('use_matched_input', True))}")
    return model, hp, run_dir


def load_gr_tfilm(run_name: str | None = DEFAULT_GR_TFILM_RUN,
                  runs_dir: str = GR_TFILM_RUNS, device: str = DEVICE):
    """Load a GR-temporal-FiLM model (``GRTFiLMDiffSSLLSTM``, 06_output/model_tfilm).

    LSTM(tvcond) for the static knobs + temporal FiLM driven by the GR curve. GR
    enters as a *conditioning* (γ/β) signal on the LSTM hidden features, NOT a
    multiplicative prior — so the amplitude-match baseline is **not** this
    model's zero-init (unlike the gain-prior / e2e models). Its
    ``forward(dry, gr, params)`` / ``reset_states`` / ``detach_states`` /
    ``cond_block_size`` match the gain-prior streaming contract, so
    ``stream_gain_prior`` drives it unchanged. Trained on the **oracle** GR
    curve. Returns (model, hparams, run_dir); eval mode on ``device``.
    """
    run_dir = _resolve_run(runs_dir, run_name, ("GRTFiLMDiffSSLLSTM",))
    with open(os.path.join(run_dir, "hparams.json")) as f:
        hp = json.load(f)
    assert hp.get("model_type") == "GRTFiLMDiffSSLLSTM", hp.get("model_type")
    m = hp["model"]
    model = GRTFiLMDiffSSLLSTM(
        num_controls=int(m["num_controls"]),
        hidden_size=int(m["hidden_size"]),
        num_layers=int(m["num_layers"]),
        tvcond_dim=int(m.get("tvcond_dim", 16)),
        cond_block_size=int(m.get("cond_block_size", 128)),
        cond_num_layers=int(m.get("cond_num_layers", 1)),
        gr_tfilm_block_size=int(m.get("gr_tfilm_block_size", 128)),
        gr_tfilm_num_layers=int(m.get("gr_tfilm_num_layers", 1)),
    )
    ckpt = latest_best_checkpoint(run_dir)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in sd.items()
                           if k.startswith("model.")})
    model.to(device).eval()
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,}-param "
          f"GRTFiLMDiffSSLLSTM  ({Path(run_dir).name} / {Path(ckpt).name})")
    return model, hp, run_dir


def load_detector(run_name: str | None = DEFAULT_DETECTOR_RUN,
                  runs_dir: str = GR_PRED_RUNS, device: str = DEVICE):
    """Load the 05_conditioning DetectorGRLSTM. Returns (model, hparams, run_dir)."""
    run_dir = _resolve_run(runs_dir, run_name, ("detector_gr_lstm",))
    with open(os.path.join(run_dir, "hparams.json")) as f:
        hp = json.load(f)
    assert hp.get("model_type") == "detector_gr_lstm", hp.get("model_type")
    m = hp["model"]
    model = DetectorGRLSTM(
        hop_size=int(hp["hop_size"]),
        sample_rate=int(hp["sample_rate"]),
        detector_taus_ms=tuple(m.get("detector_taus_ms_init", (12.0, 2.0, 40.0, 150.0))),
        detector_kernel_frames=int(m.get("detector_kernel_frames", 256)),
        hidden_size=int(m["hidden_size"]),
    )
    ckpt = latest_best_checkpoint(run_dir)
    sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in sd.items()
                           if k.startswith("model.")})
    model.to(device).eval()
    taus = [round(float(v), 1) for v in model.detector.taus_ms]
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,}-param "
          f"DetectorGRLSTM  ({Path(run_dir).name} / {Path(ckpt).name})  "
          f"learned τ={taus} ms")
    return model, hp, run_dir


def load_blackbox_gr(run_name: str | None = DEFAULT_BLACKBOX_GR_RUN,
                     runs_dir: str = GR_PRED_RUNS, ckpt_name: str | None = None,
                     device: str = DEVICE):
    """Load a frozen blackbox GR predictor (``BlackboxGRLSTM``, 05_conditioning).

    Stage 1 of the 09_end2end cascade — the local, inference-only counterpart of
    ``gr_frontend.load_frozen_gr_predictor``. Its streaming interface is
    identical to ``DetectorGRLSTM`` (``forward(dry, params, state, return_state)``,
    ``to_db``, ``hop_size``), so ``stream_detector_gr`` drives it unchanged; the
    only difference is a learned conv frontend in place of the fixed detector
    bank. Pass ``ckpt_name`` (e.g. the e2e run's ``hp['gr_frontend']['ckpt']``)
    to pin the exact checkpoint the cascade was trained against. Returns
    (model, hparams, run_dir); model frozen + eval on ``device``.
    """
    run_dir = _resolve_run(runs_dir, run_name,
                           ("blackbox_gr_lstm_film", "blackbox_gr_lstm"))
    with open(os.path.join(run_dir, "hparams.json")) as f:
        hp = json.load(f)
    assert hp.get("model_type", "").startswith("blackbox_gr_lstm"), hp.get("model_type")
    m = hp["model"]
    model = BlackboxGRLSTM(
        hop_size=int(hp["hop_size"]),
        sample_rate=int(hp["sample_rate"]),
        enc_channels=int(m["enc_channels"]),
        frontend_channels=int(m["frontend_channels"]),
        temporal_kernel=int(m["temporal_kernel"]),
        temporal_dilations=tuple(int(d) for d in m["temporal_dilations"]),
        hidden_size=int(m["hidden_size"]),
        film_order=int(m["film_order"]),
    )
    ckpt = (os.path.join(run_dir, "checkpoints", ckpt_name) if ckpt_name
            else latest_best_checkpoint(run_dir))
    sd = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in sd.items()
                           if k.startswith("model.")})
    model.to(device).eval().requires_grad_(False)
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,}-param "
          f"{hp['model_type']}  ({Path(run_dir).name} / {Path(ckpt).name})")
    return model, hp, run_dir


def load_split(run_dir: str) -> SplitManifest:
    return SplitManifest.load(os.path.join(run_dir, "split_manifest.json"))


def pairs_from_keys(keys) -> list[tuple[str, str]]:
    """split_manifest pair keys -> sorted [(song, setting), ...]."""
    return [tuple(k.split("::")) for k in sorted(keys)]


# ── dataset access ───────────────────────────────────────────────────


def pair_paths(setting: str, song: str,
               data_root: str = DATA_ROOT) -> tuple[str, str, str]:
    """(dry_wav, wet_wav, gr_curve_pt) paths for one (song, setting) pair.

    Wet is resolved per-file across the candidate setting dirs so that the
    external held-out pairs (under ``test_ground_truth/``, the v2 split's test
    set) load alongside the train/val pairs (under ``processed_ground_truth/``);
    ``wet_dir_for_setting`` alone is test-blind.
    """
    dry = os.path.join(data_root, "processed_normalized", f"{song}_UnmasteredWAV.wav")
    fname = f"{song}-exported.wav"
    wet_candidates = [
        os.path.join(data_root, "processed_ground_truth", setting, fname),
        os.path.join(data_root, "test_ground_truth", setting, fname),
        os.path.join(data_root, setting, fname),
    ]
    wet = next((w for w in wet_candidates if os.path.isfile(w)), wet_candidates[0])
    gr = os.path.join(data_root, "gr_curves", setting, f"{song}.pt")
    return dry, wet, gr


def load_pair(setting: str, song: str, start_sec: float = 0.0,
              duration_sec: float | None = None, sr: int = SR,
              data_root: str = DATA_ROOT):
    """Load (dry [1,T], wet [1,T], gr [1,T] dB) trimmed to common length."""
    dry_path, wet_path, gr_path = pair_paths(setting, song, data_root)
    total = _pair_num_frames(dry_path, wet_path, sr)
    start = int(round(start_sec * sr))
    stop = total if duration_sec is None else min(total, start + int(round(duration_sec * sr)))
    dry, wet = _read_dry_wet_segment(dry_path, wet_path, start, stop, sr)
    gr = torch.load(gr_path, map_location="cpu", weights_only=False)["gr_db"].float()
    if gr.dim() == 1:
        gr = gr.unsqueeze(0)
    gr = gr[..., start:stop]
    n = min(dry.shape[-1], wet.shape[-1], gr.shape[-1])
    return dry[..., :n], wet[..., :n], gr[..., :n]


def params_for(setting: str, device: str = DEVICE) -> torch.Tensor:
    """Normalised static knobs [1, 4] from the setting folder name."""
    return torch.tensor(normalize_setting_params(setting), dtype=torch.float32,
                        device=device).unsqueeze(0)


def knobs(threshold: float, attack_ms: float, release_s: float, ratio: float,
          device: str = DEVICE) -> torch.Tensor:
    """Normalised knob tensor [1, 4] from PHYSICAL values (for synthetic probes).

    Order per src.dsp.PARAM_ORDER = [threshold, attack, release, ratio];
    ranges per splits.DIFFSSL_PARAM_RANGES. Values outside the training grid
    are allowed — that is the point of the interpolation experiments.
    """
    phys = {"threshold": threshold, "attack": attack_ms,
            "release": release_s, "ratio": ratio}
    vals = []
    for key in PARAM_ORDER:
        lo, hi = DIFFSSL_PARAM_RANGES[key]
        vals.append((float(phys[key]) - lo) / (hi - lo))
    return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)


# ── stateful streaming (deployment-exact, mirrors the eval notebooks) ──


@torch.no_grad()
def stream_gain_prior(model, dry: torch.Tensor, gr: torch.Tensor,
                      params: torch.Tensor, chunk_sec: float = 10.0,
                      sr: int = SR, reset: bool = True,
                      device: str = DEVICE) -> torch.Tensor:
    """Stream (dry [1,T], gr [1,T] dB) through a gain-prior model -> pred [1,T].

    Chunk length snapped to the tvcond block size (exactness condition);
    reset_states() once at the start, detach_states() between chunks.
    """
    block = int(model.cond_block_size)
    chunk = max(block, (int(round(chunk_sec * sr)) // block) * block)
    n = min(dry.shape[-1], gr.shape[-1])
    if reset:
        model.reset_states()
    outs = []
    for o in range(0, n, chunk):
        d = dry[..., o:o + chunk].unsqueeze(0).to(device)
        g = gr[..., o:o + chunk].unsqueeze(0).to(device)
        outs.append(model(d, g, params).squeeze(0).cpu())
        model.detach_states()
    return torch.cat(outs, dim=-1)


@torch.no_grad()
def stream_gain_prior_parts(model, dry, gr, params, chunk_sec: float = 10.0,
                            sr: int = SR, device: str = DEVICE):
    """Like stream_gain_prior but also returns (delta_db [1,T], residual [1,T]).

    `residual` is the model's non-gain output share: additive c for the
    additive variant, (W(s)−s)+c for the waveshaper variant (return_parts
    contract shared by both models).
    """
    block = int(model.cond_block_size)
    chunk = max(block, (int(round(chunk_sec * sr)) // block) * block)
    n = min(dry.shape[-1], gr.shape[-1])
    model.reset_states()
    ys, ds, cs = [], [], []
    for o in range(0, n, chunk):
        d = dry[..., o:o + chunk].unsqueeze(0).to(device)
        g = gr[..., o:o + chunk].unsqueeze(0).to(device)
        y, delta, res = model(d, g, params, return_parts=True)
        model.detach_states()
        ys.append(y.squeeze(0).cpu())
        ds.append(delta.squeeze(0).cpu())
        cs.append(res.squeeze(0).cpu())
    return (torch.cat(ys, -1), torch.cat(ds, -1), torch.cat(cs, -1))


@torch.no_grad()
def stream_detector_gr(model, dry: torch.Tensor, params: torch.Tensor,
                       chunk_sec: float = 10.0, sr: int = SR,
                       sample_len: int | None = None, state=None,
                       device: str = DEVICE) -> torch.Tensor:
    """Stream dry [1,T] through a frame-rate GR predictor with explicit state carry.

    Works unchanged for either stage-1 model — ``DetectorGRLSTM`` (fixed
    detector bank) or ``BlackboxGRLSTM`` (learned conv frontend) — since both
    share the ``forward(dry, params, state, return_state)`` / ``to_db`` /
    ``hop_size`` streaming contract. Returns the frame-rate GR curve
    [1, T//hop] in dB, or — if `sample_len` is given — the clamped,
    sample-rate-interpolated curve [1, sample_len] (`to_db`, the exact format
    the gain-prior stage consumes).
    """
    hop = int(model.hop_size)
    chunk = max(hop, (int(round(chunk_sec * sr)) // hop) * hop)
    n = dry.shape[-1] - dry.shape[-1] % hop
    outs = []
    for o in range(0, n, chunk):
        d = dry[..., o:o + chunk].unsqueeze(0).to(device)
        gr, state = model(d, params, state, return_state=True)
        outs.append(gr.squeeze(0).cpu())
    gr = torch.cat(outs, dim=-1)
    if sample_len is not None:
        gr = model.to_db(gr.unsqueeze(0), sample_len=min(sample_len, gr.shape[-1] * hop))
        gr = gr.squeeze(0)
        if gr.shape[-1] < sample_len:      # pad sub-frame tail with last value
            pad = gr[..., -1:].expand(1, sample_len - gr.shape[-1])
            gr = torch.cat([gr, pad], dim=-1)
    return gr


# ═════════════════════════════════════════════════════════════════════
# 9-column metric engine — ported verbatim from the 05/06 eval notebooks
# (origin: 02b_sota_training/eval_lstm_diffssl_tvc.ipynb; MR-STFT and ESR
# verified there against nablafx.evaluation / auraloss to <1e-3 / <1e-4).
# ═════════════════════════════════════════════════════════════════════

from scipy.signal import bilinear, lfilter  # noqa: E402

EPS = 1e-10
MR_STE_WINDOWS = (256, 1024, 4096)
MR_NRMSE_WINDOWS = (512, 1024, 2048)
STFT_SIZES = (512, 1024, 2048)
_MRSTFT_RES = [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]

METRIC_COLS = [
    "GR MAE (dB)", "MR-STE",                          # GR stage
    "MR-STFT", "ESR (A-wt)",                          # colour stage
    "MAE (L1)", "MSE (L2)", "EDC", "M_NRMSE", "M_SF",  # standard
]


def _moving_meansq(x_sq, W, centered):
    """Sliding mean of x² via prefix sums. centered=symmetric, else causal."""
    n = x_sq.shape[-1]
    c = np.empty(n + 1)
    c[0] = 0.0
    np.cumsum(x_sq, out=c[1:])
    i = np.arange(n)
    if centered:
        lo = i - (W // 2)
        hi = lo + W
    else:                                   # causal == src.dsp_torch.gain_reduction_db
        hi = i + 1
        lo = hi - W
    lo = np.clip(lo, 0, n)
    hi = np.clip(hi, 0, n)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _to_db(x):
    return 20.0 * np.log10(np.maximum(x, EPS))


def _esr(pred, tgt):
    return float(np.sum((tgt - pred) ** 2) / (np.sum(tgt * tgt) + 1e-8))


def _a_weighting_ba(fs):
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    A1000 = 1.9997
    nums = [(2 * np.pi * f4) ** 2 * 10 ** (A1000 / 20.0), 0, 0, 0, 0]
    dens = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                      [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    return bilinear(nums, dens, fs)


_AW_CACHE: dict[int, tuple] = {}


def _aw(sr):
    if sr not in _AW_CACHE:
        _AW_CACHE[sr] = _a_weighting_ba(sr)
    return _AW_CACHE[sr]


def reference_gr_db(dry, wet, W=RMS_WINDOW):
    """Reference GR trajectory = causal RMS GR (matches src.dsp_torch)."""
    return (_to_db(np.sqrt(_moving_meansq(wet * wet, W, centered=False)))
            - _to_db(np.sqrt(_moving_meansq(dry * dry, W, centered=False))))


def mr_ste(pred, wet, windows=MR_STE_WINDOWS):
    pe, we = pred * pred, wet * wet
    total = 0.0
    for W in windows:
        ep = _moving_meansq(pe, W, centered=True)
        et = _moving_meansq(we, W, centered=True)
        total += np.sum(np.abs(ep - et)) / (np.sum(np.abs(et)) + 1e-8)
    return float(total / len(windows))


def mr_nrmse(pred, wet, windows=MR_NRMSE_WINDOWS):
    total = 0.0
    for W in windows:
        rp = np.sqrt(_moving_meansq(pred * pred, W, centered=False))
        rt = np.sqrt(_moving_meansq(wet * wet, W, centered=False))
        total += np.sqrt(np.mean((rt - rp) ** 2)) / (np.sqrt(np.mean(rt * rt)) + 1e-8)
    return float(total / len(windows))


_HANN: dict[int, torch.Tensor] = {}


def _win(n):
    if n not in _HANN:
        _HANN[n] = torch.hann_window(n)
    return _HANN[n]


def _stft_mag_pow(x2d, n_fft, hop, win):
    st = torch.stft(x2d, n_fft=n_fft, hop_length=hop, win_length=win,
                    window=_win(win), return_complex=True)
    return torch.sqrt(torch.clamp(st.real ** 2 + st.imag ** 2, min=1e-8))


def _stft_mag_fft(x2d, n_fft):
    return torch.stft(x2d, n_fft=n_fft, hop_length=n_fft // 4,
                      window=_win(n_fft), return_complex=True).abs()


@torch.no_grad()
def fft_metrics(pred1d: torch.Tensor, wet1d: torch.Tensor) -> dict:
    """auraloss MR-STFT + src.losses {M_SF, EDC} for one (pred, wet) pair."""
    P1, T1 = pred1d.unsqueeze(0), wet1d.unsqueeze(0)
    out = {}
    acc = 0.0
    for n_fft, hop, win in _MRSTFT_RES:
        P = _stft_mag_pow(P1, n_fft, hop, win)[0]
        T = _stft_mag_pow(T1, n_fft, hop, win)[0]
        sc = torch.linalg.norm(T - P) / torch.linalg.norm(T)
        lm = (torch.log(P) - torch.log(T)).abs().mean()
        acc += float(sc + lm)
    out["MR-STFT"] = acc / len(_MRSTFT_RES)

    sf = 0.0
    for n_fft in STFT_SIZES:
        P = _stft_mag_fft(P1, n_fft)[0]
        T = _stft_mag_fft(T1, n_fft)[0]
        pf, tf = torch.diff(P, dim=-1), torch.diff(T, dim=-1)
        sf += float((tf - pf).abs().mean() / (tf.abs().mean() + 1e-8))
    out["M_SF"] = sf / len(STFT_SIZES)

    pe = torch.flip(torch.cumsum(torch.flip(P1 ** 2, dims=(-1,)), dim=-1), dims=(-1,))
    te = torch.flip(torch.cumsum(torch.flip(T1 ** 2, dims=(-1,)), dim=-1), dims=(-1,))
    pe = pe / (pe[..., :1] + 1e-8)
    te = te / (te[..., :1] + 1e-8)
    out["EDC"] = float((10 * torch.log10(te.clamp(min=1e-8))
                        - 10 * torch.log10(pe.clamp(min=1e-8))).abs().mean())
    return out


def chunk_metrics(dry: np.ndarray, pred: np.ndarray, wet: np.ndarray,
                  sr: int = SR) -> dict:
    """The 9 metric columns for one (dry, pred, wet) chunk (float64 1-D)."""
    gr_tgt = np.clip(reference_gr_db(dry, wet), GR_DB_MIN, GR_DB_MAX)
    gr_pred = np.clip(reference_gr_db(dry, pred), GR_DB_MIN, GR_DB_MAX)
    aw_b, aw_a = _aw(sr)
    row = {
        "GR MAE (dB)": float(np.mean(np.abs(gr_pred - gr_tgt))),
        "MR-STE": mr_ste(pred, wet),
        "ESR (A-wt)": _esr(lfilter(aw_b, aw_a, pred), lfilter(aw_b, aw_a, wet)),
        "MAE (L1)": float(np.mean(np.abs(pred - wet))),
        "MSE (L2)": float(np.mean((pred - wet) ** 2)),
        "M_NRMSE": mr_nrmse(pred, wet),
    }
    row.update(fft_metrics(torch.from_numpy(pred).float(),
                           torch.from_numpy(wet).float()))
    return row


def score_signal(dry: torch.Tensor, pred: torch.Tensor, wet: torch.Tensor,
                 sr: int = SR, chunk_sec: float = 10.0) -> list[dict]:
    """Split full-length [1,T] tensors into chunks and score each -> metric rows.

    Chunking bounds FFT memory and matches the eval notebooks' protocol;
    aggregate with `aggregate_rows` (duration-weighted).
    """
    n = min(dry.shape[-1], pred.shape[-1], wet.shape[-1])
    chunk = int(round(chunk_sec * sr))
    min_frames = max(STFT_SIZES)
    d = dry[0, :n].double().numpy()
    p = pred[0, :n].double().numpy()
    w = wet[0, :n].double().numpy()
    rows = []
    for ci, o in enumerate(range(0, n, chunk), start=1):
        stop = min(o + chunk, n)
        if stop - o < min_frames:
            continue
        rows.append({"Chunk": ci, "Frames": stop - o,
                     **chunk_metrics(d[o:stop], p[o:stop], w[o:stop], sr)})
    return rows


def aggregate_rows(rows: list[dict], cols=None) -> dict:
    """Frame-weighted average of metric rows (same as the eval notebooks)."""
    return _weighted_average_metric_rows(rows, cols or METRIC_COLS)


# ── probe signals (detector characterisation / harmonic analysis) ────


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(a) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(np.asarray(a, dtype=np.float64), EPS))


def sine(freq: float, dur_sec: float, level_dbfs: float = -12.0,
         sr: int = SR, fade_ms: float = 5.0) -> torch.Tensor:
    """Sine tone [1, T]; `level_dbfs` is the PEAK amplitude in dBFS."""
    t = torch.arange(int(dur_sec * sr), dtype=torch.float32) / sr
    x = db_to_amp(level_dbfs) * torch.sin(2 * np.pi * freq * t)
    nf = int(fade_ms / 1000 * sr)
    if nf > 0:
        ramp = torch.linspace(0.0, 1.0, nf)
        x[:nf] *= ramp
        x[-nf:] *= ramp.flip(0)
    return x.unsqueeze(0)


def level_step(freq: float, level_a_dbfs: float, level_b_dbfs: float,
               dur_a_sec: float, dur_b_sec: float, sr: int = SR) -> torch.Tensor:
    """Tone whose PEAK level steps A→B at t=dur_a (attack/release probes). [1,T]."""
    t = torch.arange(int((dur_a_sec + dur_b_sec) * sr), dtype=torch.float32) / sr
    carrier = torch.sin(2 * np.pi * freq * t)
    amp = torch.where(t < dur_a_sec,
                      torch.tensor(db_to_amp(level_a_dbfs)),
                      torch.tensor(db_to_amp(level_b_dbfs)))
    return (carrier * amp).unsqueeze(0)


def noise(dur_sec: float, level_dbfs_rms: float = -15.0, color: str = "white",
          sr: int = SR, seed: int = 0) -> torch.Tensor:
    """White or pink noise [1, T] calibrated to an RMS level in dBFS."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(int(dur_sec * sr), generator=g)
    if color == "pink":                      # -3 dB/oct via 1/sqrt(f) spectral shaping
        X = torch.fft.rfft(x)
        f = torch.fft.rfftfreq(x.shape[-1], 1 / sr)
        X = X / torch.sqrt(torch.clamp(f, min=1.0))
        x = torch.fft.irfft(X, n=x.shape[-1])
    x = x / x.pow(2).mean().sqrt() * db_to_amp(level_dbfs_rms)
    return x.unsqueeze(0)


def rms_dbfs(x: torch.Tensor) -> float:
    return float(10 * torch.log10(x.pow(2).mean() + 1e-12))


def fit_t63(t: np.ndarray, y: np.ndarray, y0: float, y1: float) -> float:
    """Time constant via the 63.2 % crossing of a step response y0→y1.

    Robust to non-exponential tails: returns the first time y crosses
    y0 + 0.632·(y1−y0). NaN if it never crosses.
    """
    target = y0 + 0.632 * (y1 - y0)
    rising = y1 > y0
    idx = np.argmax(y >= target) if rising else np.argmax(y <= target)
    crossed = (y >= target) if rising else (y <= target)
    if not crossed.any():
        return float("nan")
    return float(t[idx])


def harmonic_levels(y: np.ndarray, sr: int, f0: float, n_harm: int = 10,
                    n_fft: int = 2 ** 16) -> np.ndarray:
    """Harmonic magnitudes (linear amplitude) of a steady tone from its tail.

    Uses the last n_fft samples with a Hann window; harmonic k is read at the
    nearest bin to k·f0 (max over ±2 bins to absorb off-bin leakage). Pick f0
    on-bin (f0 = round(f·n_fft/sr)·sr/n_fft) for clean readings.
    """
    assert y.shape[-1] >= n_fft, f"need ≥{n_fft} samples, got {y.shape[-1]}"
    seg = y[-n_fft:] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(seg)) / (n_fft / 4)      # hann coherent-gain corr.
    out = []
    for k in range(1, n_harm + 1):
        b = int(round(k * f0 * n_fft / sr))
        if b >= len(spec):
            out.append(0.0)
            continue
        lo, hi = max(0, b - 2), min(len(spec), b + 3)
        out.append(float(spec[lo:hi].max()))
    return np.asarray(out)


def on_bin_freq(f: float, sr: int = SR, n_fft: int = 2 ** 16) -> float:
    """Snap a frequency onto an FFT bin so harmonic_levels reads it cleanly."""
    return round(f * n_fft / sr) * sr / n_fft


def waveshaper_harmonics(ws_fn, amplitude: float, n_harm: int = 10,
                         n_theta: int = 8192) -> np.ndarray:
    """Exact harmonic series of a MEMORYLESS curve at drive `amplitude`.

    Feeds s(θ)=A·cos(θ) through ws_fn (callable on a [1,1,N] tensor) and
    Fourier-analyses over one period — harmonic k amplitude = 2|Y_k|/N.
    """
    theta = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    s = torch.tensor(amplitude * np.cos(theta), dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        y = ws_fn(s).squeeze().numpy().astype(np.float64)
    Y = np.fft.rfft(y) / n_theta
    mags = 2 * np.abs(Y[1:n_harm + 1])
    return mags


# ── onset detection (transient analysis) ─────────────────────────────


def detect_onsets(x: torch.Tensor, sr: int = SR, hop: int = 256,
                  rise_db: float = 8.0, floor_db: float = -45.0,
                  min_sep_sec: float = 0.4, lookback_frames: int = 4) -> np.ndarray:
    """Energy-rise onset detector on the DRY signal -> onset sample indices.

    A frame is an onset candidate if its dB energy exceeds the energy
    `lookback_frames` earlier by `rise_db` and sits above `floor_db`.
    Candidates within `min_sep_sec` collapse to the strongest rise.
    Deliberately simple — we need alignment anchors, not MIR-grade recall.
    """
    e = torch.nn.functional.avg_pool1d((x * x).unsqueeze(0), hop).squeeze().numpy()
    e_db = 10 * np.log10(e + 1e-12)
    rise = np.zeros_like(e_db)
    rise[lookback_frames:] = e_db[lookback_frames:] - e_db[:-lookback_frames]
    cand = np.where((rise > rise_db) & (e_db > floor_db))[0]
    if len(cand) == 0:
        return np.array([], dtype=int)
    min_sep = int(min_sep_sec * sr / hop)
    onsets, group = [], [cand[0]]
    for c in cand[1:]:
        if c - group[-1] <= min_sep:
            group.append(c)
        else:
            onsets.append(group[int(np.argmax(rise[group]))])
            group = [c]
    onsets.append(group[int(np.argmax(rise[group]))])
    return np.asarray(onsets) * hop


def short_env_db(x: torch.Tensor, win: int = 128) -> np.ndarray:
    """Causal short-window RMS envelope in dB of a [1,T] signal (1-D out)."""
    xx = x[0].double().numpy()
    return _to_db(np.sqrt(_moving_meansq(xx * xx, win, centered=False)))
