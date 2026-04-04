import os
import sys
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nablafx"))

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root-Mean-Square Error (RMSE) to measure energy deviation.
    
    Using the paper's formula: RMSE = sqrt( 1/N sum_n (|y_n|^2 - |\hat{y}_n|^2) )
    We add absolute value to ensure numerical stability for the square root.
    """
    return torch.sqrt(torch.mean(torch.abs(target**2 - pred**2))).item()


def spectral_flux_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Spectral Flux Error (SFE).
    
    Computed on consecutive, overlapping windows using n_fft=2048 and 75% overlap.
    Returns the L1 norm divided by N.
    """
    n_fft = 2048
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=pred.device)

    pred_spec = torch.stft(
        pred.squeeze(),
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        return_complex=True,
    ).abs()

    tgt_spec = torch.stft(
        target.squeeze(),
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        return_complex=True,
    ).abs()

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
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)
        pred_spec = torch.stft(
            pred.squeeze(),
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
        ).abs()
        tgt_spec = torch.stft(
            target.squeeze(),
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
        ).abs()
        
        num = torch.sum(torch.abs(tgt_spec - pred_spec))
        den = torch.sum(tgt_spec) + 1e-8
        total += (num / den).item()

    return total / len(fft_sizes)


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error (MSE)."""
    return torch.mean((target - pred) ** 2).item()


def esr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Error-to-Signal Ratio (ESR)."""
    return (torch.sum((target - pred) ** 2) / (torch.sum(target ** 2) + 1e-8)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error (MAE)."""
    return torch.mean(torch.abs(target - pred)).item()


def compute_all_losses(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Unique function that calculates just the 6 explicitly requested losses.
    
    Zero-pads to 2048 when necessary to satisfy SFE / M-STFTE window sizes.
    """
    min_len = 2048
    if a.size(-1) < min_len:
        pad_len = min_len - a.size(-1)
        a_pad = torch.nn.functional.pad(a, (0, pad_len))
        b_pad = torch.nn.functional.pad(b, (0, pad_len))
    else:
        a_pad = a
        b_pad = b

    return {
        "MSE": mse(a, b),
        "MAE": mae(a, b),
        "ESR": esr(a, b),
        "RMSE": rmse(a, b),
        "SFE": spectral_flux_error(a_pad, b_pad),
        "M-STFTE": m_stfte(a_pad, b_pad),
    }


def evaluate_to_dataframe(sig_wet, output_eval, loss_module, compressor_name, print_results=False):
    """
    Evaluates recreated signals against a target wet signal and returns a DataFrame.
    Optionally prints the formatted comparison table.
    
    Args:
        sig_wet: The target wet signal array.
        output_eval (dict): Dictionary mapping evaluation method names to recreated signals.
        loss_module: Module or object containing the `compute_all_losses` method.
        compressor_name (str): Name of the compressor being evaluated.
        print_results (bool): If True, prints a formatted table of the computed losses.
        
    Returns:
        pd.DataFrame: A DataFrame containing the compressor name, evaluation method, and computed losses.
    """
    s_wet = torch.as_tensor(sig_wet, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    rows = []
    
    # Optional setup for printing
    if print_results and output_eval:
        # Get loss names by previewing the first result
        first_recreated = next(iter(output_eval.values()))
        s_first = torch.as_tensor(first_recreated, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        first_losses = loss_module.compute_all_losses(s_first, s_wet)
        loss_names = list(first_losses.keys())
        
        label_width = max(len(name) for name in output_eval.keys()) + 2
        col_width = 10
        
        header = f"\n{compressor_name:<{label_width}}  " + "  ".join(f"{n:>{col_width}}" for n in loss_names)
        print(header)
        print("─" * len(header))
    
    for eval_name, recreated_sig in output_eval.items():
        s_recreated = torch.as_tensor(recreated_sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Calculate losses (returns a dict of loss_name: loss_value)
        losses = loss_module.compute_all_losses(s_recreated, s_wet)
        
        # Extract scalar values from torch tensors if needed
        parsed_losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        
        # Construct the row data
        row_data = {
            'Compressor': compressor_name,
            'Method': eval_name,
        }
        row_data.update(parsed_losses)
        
        rows.append(row_data)
        
        # Optional printing per row
        if print_results:
            vals = "  ".join(f"{parsed_losses[n]:>{col_width}.6f}" for n in loss_names)
            print(f"{eval_name:<{label_width}}  {vals}")
        
    # Create and return the pandas DataFrame
    return pd.DataFrame(rows)
