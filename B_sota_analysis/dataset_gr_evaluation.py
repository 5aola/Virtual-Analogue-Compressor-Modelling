#!/usr/bin/env python
"""
Dataset Gain Reduction Evaluation Script.

Calculates the average gain reduction and its standard deviation
for each input-output pair in the given compressor dataset.

Visualizes how these values change across different compressor settings
and shows the variation across different songs.
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_io import load_audio
from src.dsp import calc_gain_reduction

GT_ROOT = "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_ground_truth"
INPUT_ROOT = "/Volumes/Saola's Drive/thesis/data/Diff-SSL-G-Comp/processed_normalized"

SETTING_RE = re.compile(
    r"^threshold_(?P<threshold>-?\d+(?:\.\d+)?)"
    r"_attack_(?P<attack>\d+(?:\.\d+)?)"
    r"_release_(?P<release>\d+(?:\.\d+)?)"
    r"_ratio_(?P<ratio>\d+(?:\.\d+)?)$"
)


def discover_settings(gt_root: str) -> list[dict]:
    """Scan gt_root for setting folders and return parsed parameter dicts."""
    settings = []
    if not os.path.exists(gt_root):
        print(f"Warning: GT_ROOT {gt_root} does not exist.")
        return settings
        
    for name in sorted(os.listdir(gt_root)):
        full = os.path.join(gt_root, name)
        if not os.path.isdir(full):
            continue
        m = SETTING_RE.match(name)
        if m is None:
            continue
            
        settings.append({
            "folder_name": name,
            "path": full,
            "threshold": float(m.group("threshold")),
            "attack": float(m.group("attack")),
            "release": float(m.group("release")),
            "ratio": float(m.group("ratio")),
        })
    return settings


def get_all_songs(input_root: str, max_songs: int = None) -> list[str]:
    """Get a list of all unmastered wav sizes."""
    if not os.path.exists(input_root):
        print(f"Warning: INPUT_ROOT {input_root} does not exist.")
        return []
        
    songs = [f for f in sorted(os.listdir(input_root)) if f.endswith(".wav")]
    if max_songs is not None and max_songs > 0:
        songs = songs[:max_songs]
    return songs


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dataset Gain Reduction")
    parser.add_argument("--gt-root", default=GT_ROOT, help="Ground truth root directory")
    parser.add_argument("--input-root", default=INPUT_ROOT, help="Input (unmastered) root directory")
    parser.add_argument("--output-dir", default="B_sota_analysis/eval_output/dataset_gr_eval", help="Output directory for CSV and plots")
    parser.add_argument("--max-songs", type=int, default=None, help="Maximum number of songs to evaluate (for quick testing)")
    parser.add_argument("--window-size", type=int, default=64, help="Window size for RMS/Peak calculation")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate")
    parser.add_argument("--from-csv", action="store_true", help="Skip evaluation and plot from existing CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "dataset_gr_evaluation.csv")

    results = []

    if args.from_csv and os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        print("Discovering settings and songs...")
        settings = discover_settings(args.gt_root)
        songs = get_all_songs(args.input_root, args.max_songs)
        
        print(f"Found {len(settings)} settings and {len(songs)} songs.")
        
        if not settings or not songs:
            print("Error: Could not find settings or songs. Check your directories.")
            return

        # Main evaluation loop
        for song_filename in tqdm(songs, desc="Processing songs"):
            song_base = song_filename.split("_")[0]
            target_filename = f"{song_base}-exported.wav"
            input_path = os.path.join(args.input_root, song_filename)
            
            # Check which settings actually have this song
            valid_settings = []
            for stg in settings:
                target_path = os.path.join(stg["path"], target_filename)
                if os.path.exists(target_path):
                    valid_settings.append((stg, target_path))
                    
            if not valid_settings:
                # Skip loading unmastered audio if there are no targets for this song
                continue
                
            # Load input audio only once if it's needed
            try:
                _, x_np = load_audio(input_path, sr=args.sample_rate)
            except Exception as e:
                print(f"Failed to load input {input_path}: {e}")
                continue

            for stg, target_path in valid_settings:
                try:
                    _, y_np = load_audio(target_path, sr=args.sample_rate)
                except Exception as e:
                    print(f"Failed to load target {target_path}: {e}")
                    continue

                # Ensure same length
                min_len = min(len(x_np), len(y_np))
                x_trim = x_np[:min_len]
                y_trim = y_np[:min_len]

                # Calculate gain reduction (dB difference per frame)
                # gr_db is y_rms - x_rms, so it will be negative for gain reduction
                gr_db = calc_gain_reduction(x_trim, y_trim, window_size=args.window_size)
                
                # Since compressors *reduce* gain, average GR is typically negative.
                # We'll store it as is (or flip it to positive if preferred, but negative is standard dB measure)
                mean_gr = np.mean(gr_db)
                std_gr = np.std(gr_db)
                min_gr = np.min(gr_db)
                max_gr = np.max(gr_db)

                results.append({
                    "Song": song_base,
                    "Setting": stg["folder_name"],
                    "Threshold": stg["threshold"],
                    "Attack": stg["attack"],
                    "Release": stg["release"],
                    "Ratio": stg["ratio"],
                    "Mean_GR_dB": mean_gr,
                    "Std_GR_dB": std_gr,
                    "Min_GR_dB": min_gr,
                    "Max_GR_dB": max_gr
                })

        if not results:
            print("No paired results found.")
            return

        df = pd.DataFrame(results)
        
        # Save CSV
        df.to_csv(csv_path, index=False)
        print(f"\nSaved numerical results to {csv_path}")

    # Create visualization
    print("Generating plots...")
    
    # Create a unified setting label that encodes strictness (Threshold & Ratio)
    # We can create a string label for the categorical axis
    # Sorting by a logical order: lowest threshold (strongest comp) to highest, then ratio
    df = df.sort_values(by=["Threshold", "Ratio"], ascending=[True, False])
    
    # Create a nice label for plots focusing on Threshold and Ratio
    df["Setting_Label"] = df.apply(lambda row: f"Thr={row['Threshold']} Ratio={row['Ratio']}", axis=1)

    # Plot 1: Mean GR via stripplot to show individual songs
    plt.figure(figsize=(16, 8))
    sns.stripplot(
        data=df, 
        x="Setting_Label", 
        y="Mean_GR_dB", 
        hue="Song", 
        jitter=True, 
        alpha=0.7, 
        size=6,
        palette="tab20"
    )
    
    # Add boxplot overlay for statistical summary
    sns.boxplot(
        data=df,
        x="Setting_Label",
        y="Mean_GR_dB",
        color="white",
        width=0.4,
        fliersize=0,
        boxprops={'alpha': 0.3}
    )

    plt.title("Expected Gain Reduction per Setting (Across Songs)", fontsize=16)
    plt.xlabel("Compressor Setting", fontsize=12)
    plt.ylabel("Mean Gain Reduction (dB)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Limit legend items if too many songs
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 10:
        plt.legend(handles[:10], labels[:10], title='Song (Subset)', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='Song', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "mean_gr_by_setting.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Mean GR plot to {plot_path}")

    # Plot 2: GR Std Dev (Dynamics variation)
    plt.figure(figsize=(16, 8))
    sns.stripplot(
        data=df, 
        x="Setting_Label", 
        y="Std_GR_dB", 
        hue="Song", 
        jitter=True, 
        alpha=0.7, 
        size=6,
        palette="tab20"
    )
    
    sns.boxplot(
        data=df,
        x="Setting_Label",
        y="Std_GR_dB",
        color="white",
        width=0.4,
        fliersize=0,
        boxprops={'alpha': 0.3}
    )

    plt.title("Gain Reduction Standard Deviation per Setting (Across Songs)", fontsize=16)
    plt.xlabel("Compressor Setting", fontsize=12)
    plt.ylabel("Standard Deviation of Gain Reduction (dB)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Limit legend items if too many songs
    if len(handles) > 10:
        plt.legend(handles[:10], labels[:10], title='Song (Subset)', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title='Song', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot2_path = os.path.join(args.output_dir, "std_gr_by_setting.png")
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Std GR plot to {plot2_path}")

    print("Done!")

if __name__ == "__main__":
    main()
