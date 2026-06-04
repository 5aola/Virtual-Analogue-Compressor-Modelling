"""
Evaluation loss / metric functions for compressor modelling.

All functions accept PyTorch tensors and return Python floats.
"""

import pandas as pd
import torch
import torch.nn.functional as F


def _as_bct(x: torch.Tensor) -> torch.Tensor:
    """Return audio as [batch, channels, time]."""
    if x.ndim == 1:
        return x.view(1, 1, -1)
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 1-D, 2-D, or 3-D audio tensor, got {x.ndim}-D")


def _pad_to_min_len(x: torch.Tensor, min_len: int) -> torch.Tensor:
    if x.size(-1) >= min_len:
        return x
    return F.pad(x, (0, min_len - x.size(-1)))


def _stft_mag(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    x = _pad_to_min_len(_as_bct(x).float(), n_fft)
    flat = x.reshape(-1, x.shape[-1])
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    return torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        window=window,
        return_complex=True,
    ).abs()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root-Mean-Square Error (RMSE) to measure energy deviation.

    Using the paper's formula: RMSE = sqrt( 1/N sum_n (|y_n|^2 - |hat{y}_n|^2) )
    We add absolute value to ensure numerical stability for the square root.
    """
    return torch.sqrt(torch.mean(torch.abs(target**2 - pred**2))).item()


def spectral_flux_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Spectral Flux Error (SFE).

    Computed on consecutive, overlapping windows using n_fft=2048 and 75% overlap.
    Returns the L1 norm divided by N.
    """
    n_fft = 2048
    pred_spec = _stft_mag(pred, n_fft)
    tgt_spec = _stft_mag(target, n_fft)

    pred_flux = torch.diff(pred_spec, dim=-1)
    tgt_flux = torch.diff(tgt_spec, dim=-1)

    return torch.mean(torch.abs(tgt_flux - pred_flux)).item()


def m_stfte(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Multi-resolution Short-Time Fourier Transform Error (M-STFTE).

    Evaluates L1 distance between magnitude spectra across resolutions
    (512, 1024, 2048) with 75% overlap, normalized by target magnitude.
    """
    fft_sizes = (512, 1024, 2048)
    total = 0.0
    for n_fft in fft_sizes:
        pred_spec = _stft_mag(pred, n_fft)
        tgt_spec = _stft_mag(target, n_fft)

        num = torch.sum(torch.abs(tgt_spec - pred_spec))
        den = torch.sum(tgt_spec) + 1e-8
        total += (num / den).item()

    return total / len(fft_sizes)


def edc(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Energy-decay-curve error in dB."""
    pred = _as_bct(pred).float()
    target = _as_bct(target).float()

    pred_edc = torch.flip(torch.cumsum(torch.flip(pred**2, dims=(-1,)), dim=-1), dims=(-1,))
    target_edc = torch.flip(
        torch.cumsum(torch.flip(target**2, dims=(-1,)), dim=-1), dims=(-1,)
    )

    pred_edc = pred_edc / (pred_edc[..., :1] + 1e-8)
    target_edc = target_edc / (target_edc[..., :1] + 1e-8)

    pred_db = 10.0 * torch.log10(pred_edc.clamp(min=1e-8))
    target_db = 10.0 * torch.log10(target_edc.clamp(min=1e-8))
    return torch.mean(torch.abs(target_db - pred_db)).item()


def _moving_rms(x: torch.Tensor, window_size: int) -> torch.Tensor:
    x = _pad_to_min_len(_as_bct(x).float(), window_size)
    batch, channels, frames = x.shape
    flat = x.reshape(batch * channels, 1, frames)
    kernel = torch.ones(1, 1, window_size, device=x.device, dtype=x.dtype) / window_size
    rms_sq = F.conv1d(flat**2, kernel, padding=window_size - 1)[..., :frames]
    return torch.sqrt(rms_sq.clamp(min=1e-10)).reshape(batch, channels, frames)


def multi_resolution_nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean normalized RMS-envelope error over multiple window sizes."""
    window_sizes = (512, 1024, 2048)
    total = 0.0
    for window_size in window_sizes:
        pred_rms = _moving_rms(pred, window_size)
        target_rms = _moving_rms(target, window_size)
        err = torch.sqrt(torch.mean((target_rms - pred_rms) ** 2))
        denom = torch.sqrt(torch.mean(target_rms**2)) + 1e-8
        total += (err / denom).item()
    return total / len(window_sizes)


def multi_resolution_spectral_flux_error(
    pred: torch.Tensor, target: torch.Tensor
) -> float:
    """Mean normalized spectral-flux error over multiple STFT resolutions."""
    fft_sizes = (512, 1024, 2048)
    total = 0.0
    for n_fft in fft_sizes:
        pred_spec = _stft_mag(pred, n_fft)
        target_spec = _stft_mag(target, n_fft)
        pred_flux = torch.diff(pred_spec, dim=-1)
        target_flux = torch.diff(target_spec, dim=-1)
        num = torch.mean(torch.abs(target_flux - pred_flux))
        den = torch.mean(torch.abs(target_flux)) + 1e-8
        total += (num / den).item()
    return total / len(fft_sizes)


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error (MSE)."""
    return torch.mean((target - pred) ** 2).item()


def esr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Error-to-Signal Ratio (ESR)."""
    return (torch.sum((target - pred) ** 2) / (torch.sum(target**2) + 1e-8)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error (MAE)."""
    return torch.mean(torch.abs(target - pred)).item()


def compute_all_losses(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute the shared audio-signal evaluation metrics."""
    a_pad = _pad_to_min_len(a, 2048)
    b_pad = _pad_to_min_len(b, 2048)

    return {
        "MAE (L1)": mae(a, b),
        "MSE (L2)": mse(a, b),
        "ESR": esr(a, b),
        "MR-STFT": m_stfte(a_pad, b_pad),
        "EDC": edc(a, b),
        "M_NRMSE": multi_resolution_nrmse(a, b),
        "M_SF": multi_resolution_spectral_flux_error(a_pad, b_pad),
    }


def evaluate_to_dataframe(
    sig_wet, output_eval, compressor_name, print_results=False
):
    """
    Evaluate recreated signals against a target wet signal and return a DataFrame.

    Args:
        sig_wet: The target wet signal array.
        output_eval (dict): Dictionary mapping evaluation method names to recreated signals.
        compressor_name (str): Name of the compressor being evaluated.
        print_results (bool): If True, prints a formatted table of the computed losses.

    Returns:
        pd.DataFrame
    """
    s_wet = torch.as_tensor(sig_wet, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    rows = []

    if print_results and output_eval:
        first_recreated = next(iter(output_eval.values()))
        s_first = (
            torch.as_tensor(first_recreated, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        first_losses = compute_all_losses(s_first, s_wet)
        loss_names = list(first_losses.keys())

        label_width = max(len(name) for name in output_eval.keys()) + 2
        col_width = 10

        header = f"\n{compressor_name:<{label_width}}  " + "  ".join(
            f"{n:>{col_width}}" for n in loss_names
        )
        print(header)
        print("─" * len(header))

    for eval_name, recreated_sig in output_eval.items():
        s_recreated = (
            torch.as_tensor(recreated_sig, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        losses = compute_all_losses(s_recreated, s_wet)

        parsed_losses = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in losses.items()
        }

        row_data = {
            "Compressor": compressor_name,
            "Method": eval_name,
        }
        row_data.update(parsed_losses)

        rows.append(row_data)

        if print_results:
            vals = "  ".join(
                f"{parsed_losses[n]:>{col_width}.6f}" for n in loss_names
            )
            print(f"{eval_name:<{label_width}}  {vals}")

    return pd.DataFrame(rows)
