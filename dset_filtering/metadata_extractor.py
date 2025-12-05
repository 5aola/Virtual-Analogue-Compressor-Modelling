"""
SSL G-Comp Dataset Loader

Builds a pandas DataFrame with dry/wet audio pairs and parsed compressor settings.

Dataset structure:
    data_root/
    ├── processed_normalized/           # Dry files (x)
    │   ├── SongName_UnmasteredWAV.wav
    │   └── ...
    └── processed_ground_truth/          # Wet files (y)
        ├── _threshold_-12_attack_1_release_0.1_ratio_2/
        │   ├── SongName-exported.wav
        │   └── ...
        └── threshold_12_attack_30_release_0.8_ratio_2/
            └── ...
"""

import os
import re
import glob
import pandas as pd
import torch
import torchaudio
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple


def parse_settings_from_folder_name(folder_name: str) -> dict:
    """
    Parse compressor settings from folder name.

    Expected format: '_threshold_-12_attack_1_release_0.1_ratio_2' or similar

    Args:
        folder_name: Name of the settings folder

    Returns:
        Dictionary with parsed settings (threshold, attack, release, ratio)
    """
    settings = {
        "threshold": None,
        "attack": None,
        "release": None,
    }

    # Pattern to match: paramname_value pairs
    # Handle negative values with optional minus sign
    patterns = {
        "threshold": r"threshold_(-?\d+(?:\.\d+)?)",
        "attack": r"attack_(-?\d+(?:\.\d+)?)",
        "release": r"release_(-?\d+(?:\.\d+)?)",
        "ratio": r"ratio_(-?\d+(?:\.\d+)?)",
    }

    for param, pattern in patterns.items():
        match = re.search(pattern, folder_name, re.IGNORECASE)
        if match:
            value = match.group(1)
            settings[param] = float(value) if "." in value else int(value)

    return settings


def get_song_name_from_dry(dry_filename: str) -> str:
    """
    Extract song name from dry file.

    Example: '5thFloor_UnmasteredWAV.wav' -> '5thFloor'
    """
    basename = os.path.basename(dry_filename)
    # Remove _UnmasteredWAV.wav suffix
    song_name = basename.replace("_UnmasteredWAV.wav", "")
    return song_name


def get_song_name_from_wet(wet_filename: str) -> str:
    """
    Extract song name from wet file.

    Example: 'NosPalpitants-exported.wav' -> 'NosPalpitants'
    """
    basename = os.path.basename(wet_filename)
    # Remove -exported.wav suffix
    song_name = basename.replace("-exported.wav", "")
    return song_name


def build_dataset_dataframe(
    data_root: str,
    dry_folder: str = "processed_normalized",
    wet_folder: str = "processed_ground_truth",
    dry_suffix: str = "_UnmasteredWAV.wav",
    wet_suffix: str = "-exported.wav",
    settings_folders: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a pandas DataFrame with dry/wet audio pairs and compressor settings.

    Args:
        data_root: Root directory containing the dataset
        dry_folder: Name of folder containing dry (input) audio files
        wet_folder: Name of folder containing wet (output) audio subfolders
        dry_suffix: Suffix pattern for dry files
        wet_suffix: Suffix pattern for wet files
        settings_folders: Optional list of specific settings folders to use.
                         If None, all folders in wet_folder are used.

    Returns:
        DataFrame with columns:
            - song_name: Name of the song
            - dry_path: Full path to dry audio file
            - wet_path: Full path to wet audio file
            - settings_folder: Name of the settings folder
            - threshold: Threshold value (dB)
            - attack: Attack time
            - release: Release time
            - ratio: Compression ratio
    """
    dry_dir = os.path.join(data_root, dry_folder)
    wet_dir = os.path.join(data_root, wet_folder)

    # Validate directories exist
    if not os.path.isdir(dry_dir):
        raise ValueError(f"Dry folder not found: {dry_dir}")
    if not os.path.isdir(wet_dir):
        raise ValueError(f"Wet folder not found: {wet_dir}")

    # Get all dry files
    dry_files = glob.glob(os.path.join(dry_dir, f"*{dry_suffix}"))
    dry_files = sorted(dry_files)

    # Build lookup: song_name -> dry_path
    dry_lookup = {}
    for dry_path in dry_files:
        song_name = get_song_name_from_dry(dry_path)
        dry_lookup[song_name] = dry_path

    # Get settings folders
    if settings_folders is None:
        settings_folders = [
            d for d in os.listdir(wet_dir) if os.path.isdir(os.path.join(wet_dir, d))
        ]

    # Build dataset rows
    rows = []

    for settings_folder in sorted(settings_folders):
        settings_path = os.path.join(wet_dir, settings_folder)
        if not os.path.isdir(settings_path):
            continue

        # Parse settings from folder name
        settings = parse_settings_from_folder_name(settings_folder)

        # Get all wet files in this settings folder
        wet_files = glob.glob(os.path.join(settings_path, f"*{wet_suffix}"))

        for wet_path in sorted(wet_files):
            song_name = get_song_name_from_wet(wet_path)

            # Find matching dry file
            dry_path = dry_lookup.get(song_name)

            if dry_path is None:
                print(f"Warning: No matching dry file for {song_name}")
                continue

            row = {
                "song_name": song_name,
                "dry_path": dry_path,
                "wet_path": wet_path,
                "threshold": settings["threshold"],
                "attack": settings["attack"],
                "release": settings["release"],
                "ratio": settings["ratio"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by song name and settings for nice ordering
    if not df.empty:
        df = df.sort_values(["song_name", "threshold", "attack", "release", "ratio"])
        df = df.reset_index(drop=True)

    return df


def get_unique_settings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get unique compressor settings from the dataset.

    Args:
        df: DataFrame from build_dataset_dataframe()

    Returns:
        DataFrame with unique settings combinations
    """
    settings_cols = ["threshold", "attack", "release", "ratio", "settings_folder"]
    return df[settings_cols].drop_duplicates().reset_index(drop=True)


def get_songs_by_setting(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    attack: Optional[float] = None,
    release: Optional[float] = None,
    ratio: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter dataset by specific compressor settings.

    Args:
        df: DataFrame from build_dataset_dataframe()
        threshold: Filter by threshold (dB)
        attack: Filter by attack time
        release: Filter by release time
        ratio: Filter by compression ratio

    Returns:
        Filtered DataFrame
    """
    mask = pd.Series([True] * len(df))

    if threshold is not None:
        mask &= df["threshold"] == threshold
    if attack is not None:
        mask &= df["attack"] == attack
    if release is not None:
        mask &= df["release"] == release
    if ratio is not None:
        mask &= df["ratio"] == ratio

    return df[mask].reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> None:
    """Print a summary of the dataset."""
    print("=" * 60)
    print("SSL G-Comp Dataset Summary")
    print("=" * 60)
    print(f"Total pairs: {len(df)}")
    print(f"Unique songs: {df['song_name'].nunique()}")
    print(f"Unique settings: {df['settings_folder'].nunique()}")
    print()
    print("Settings ranges:")
    print(f"  Threshold: {df['threshold'].min()} to {df['threshold'].max()} dB")
    print(f"  Attack: {df['attack'].min()} to {df['attack'].max()}")
    print(f"  Release: {df['release'].min()} to {df['release'].max()}")
    print(f"  Ratio: {df['ratio'].min()} to {df['ratio'].max()}")
    print("=" * 60)


# -----------------------------------------------------------------------------
# PyTorch Dataset and DataModule
# -----------------------------------------------------------------------------


class SSLGCompDataset(Dataset):
    """
    PyTorch Dataset for SSL G-Comp dry/wet audio pairs.

    Args:
        df: DataFrame from build_dataset_dataframe()
        sample_length: Length of audio samples in frames. -1 for full file.
        sample_rate: Target sample rate (will resample if needed)
        return_params: Whether to return compressor parameters
        normalize_params: Whether to normalize params to [0, 1] range
        preload: Whether to preload all audio into memory
    """

    # Normalization ranges for parameters (adjust based on your data)
    PARAM_RANGES = {
        "threshold": (-20, 0),  # dB range
        "attack": (0.1, 30),  # ms range
        "release": (0.1, 1.6),  # seconds range
        "ratio": (2, 10),  # ratio range
    }

    def __init__(
        self,
        df: pd.DataFrame,
        sample_length: int = 48000,
        sample_rate: int = 48000,
        return_params: bool = True,
        normalize_params: bool = True,
        preload: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.return_params = return_params
        self.normalize_params = normalize_params
        self.preload = preload

        # Build samples list (with chunking if needed)
        self.samples = self._build_samples()

        # Preload audio if requested
        if self.preload:
            self._preload_audio()

    def _build_samples(self) -> list:
        """Build list of sample entries, with chunking for long files."""
        samples = []

        for idx, row in self.df.iterrows():
            try:
                md = torchaudio.info(row["dry_path"])
                num_frames = md.num_frames
            except Exception as e:
                print(f"Warning: Could not read {row['dry_path']}: {e}")
                continue

            # Get parameters
            params = self._get_params(row)

            if self.sample_length == -1:
                # Use whole file
                samples.append(
                    {
                        "df_idx": idx,
                        "dry_path": row["dry_path"],
                        "wet_path": row["wet_path"],
                        "params": params,
                        "offset": 0,
                        "num_frames": num_frames,
                        "dry_audio": None,
                        "wet_audio": None,
                    }
                )
            else:
                # Split into chunks
                for n in range(num_frames // self.sample_length):
                    offset = n * self.sample_length
                    samples.append(
                        {
                            "df_idx": idx,
                            "dry_path": row["dry_path"],
                            "wet_path": row["wet_path"],
                            "params": params,
                            "offset": offset,
                            "num_frames": self.sample_length,
                            "dry_audio": None,
                            "wet_audio": None,
                        }
                    )

        return samples

    def _get_params(self, row) -> torch.Tensor:
        """Extract and optionally normalize parameters."""
        params = torch.tensor(
            [
                row["threshold"],
                row["attack"],
                row["release"],
                row["ratio"],
            ],
            dtype=torch.float32,
        )

        if self.normalize_params:
            # Normalize each param to [0, 1]
            ranges = self.PARAM_RANGES
            params[0] = (params[0] - ranges["threshold"][0]) / (
                ranges["threshold"][1] - ranges["threshold"][0]
            )
            params[1] = (params[1] - ranges["attack"][0]) / (
                ranges["attack"][1] - ranges["attack"][0]
            )
            params[2] = (params[2] - ranges["release"][0]) / (
                ranges["release"][1] - ranges["release"][0]
            )
            params[3] = (params[3] - ranges["ratio"][0]) / (
                ranges["ratio"][1] - ranges["ratio"][0]
            )
            params = torch.clamp(params, 0, 1)

        return params

    def _preload_audio(self):
        """Preload all audio into memory."""
        print("Preloading audio...")
        for i, sample in enumerate(self.samples):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(self.samples)}")

            dry, _ = self._load_audio(
                sample["dry_path"],
                sample["offset"],
                sample["num_frames"] if self.sample_length != -1 else -1,
            )
            wet, _ = self._load_audio(
                sample["wet_path"],
                sample["offset"],
                sample["num_frames"] if self.sample_length != -1 else -1,
            )
            sample["dry_audio"] = dry
            sample["wet_audio"] = wet
        print("Preloading complete!")

    def _load_audio(
        self, filepath: str, frame_offset: int = 0, num_frames: int = -1
    ) -> Tuple[torch.Tensor, int]:
        """Load audio file with optional offset and length."""
        x, sr = torchaudio.load(
            filepath,
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=True,
        )

        # Resample if needed
        if sr != self.sample_rate:
            x = torchaudio.functional.resample(x, sr, self.sample_rate)

        # Convert to mono if stereo
        if x.shape[0] > 1:
            x = torch.mean(x, dim=0, keepdim=True)

        return x, sr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if self.preload:
            dry = sample["dry_audio"]
            wet = sample["wet_audio"]
        else:
            num_frames = sample["num_frames"] if self.sample_length != -1 else -1
            dry, _ = self._load_audio(sample["dry_path"], sample["offset"], num_frames)
            wet, _ = self._load_audio(sample["wet_path"], sample["offset"], num_frames)

        if self.return_params:
            return dry, wet, sample["params"]
        else:
            return dry, wet


class SSLGCompDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for SSL G-Comp dataset.

    Args:
        data_root: Root directory of the dataset
        sample_length: Audio sample length in frames
        sample_rate: Target sample rate
        train_split: Fraction of data for training
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        return_params: Whether to return compressor parameters
        normalize_params: Whether to normalize parameters
        preload: Whether to preload audio
    """

    def __init__(
        self,
        data_root: str,
        sample_length: int = 48000,
        sample_rate: int = 48000,
        train_split: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 4,
        return_params: bool = True,
        normalize_params: bool = True,
        preload: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_params = return_params
        self.normalize_params = normalize_params
        self.preload = preload

    def setup(self, stage: Optional[str] = None):
        # Build the full dataframe
        df = build_dataset_dataframe(self.data_root)
        dataset_summary(df)

        # Create full dataset
        full_dataset = SSLGCompDataset(
            df=df,
            sample_length=self.sample_length,
            sample_rate=self.sample_rate,
            return_params=self.return_params,
            normalize_params=self.normalize_params,
            preload=self.preload,
        )

        # Split into train/val
        train_size = int(len(full_dataset) * self.train_split)
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        print(f"\nTrain samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# Example usage
if __name__ == "__main__":
    # Example: adjust this path to your data location
    DATA_ROOT = "/Volumes/Production Tools/coding_projs/THESIS/data_preprocesses/data/Diff-SSL-G-Comp"

    # Build the dataset
    df = build_dataset_dataframe(DATA_ROOT)

    # Print summary
    dataset_summary(df)

    # View first few rows
    print("\nFirst 10 rows:")
    print(df.head(10))

    # Get unique settings
    print("\nUnique settings:")
    print(get_unique_settings(df))

    # Filter by specific setting
    print("\nSongs with threshold=-12, ratio=2:")
    filtered = get_songs_by_setting(df, threshold=-12, ratio=2)
    print(filtered[["song_name", "threshold", "attack", "release", "ratio"]])
