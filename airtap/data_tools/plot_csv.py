"""
plot_csv.py

loads csv time series data and plot fingertip distance over time
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

print("debug")


def plot_csv(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    if 'frame_idx' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'frame_idx' and 'label' columns")
    os.makedirs(output_dir, exist_ok=True)

    #plot
    plt.figure()
    plt.scatter(df['frame_idx'], df['label'], s=5)
    plt.xlabel('Frame Index')
    plt.ylabel('Tap Label (0 or 1)')
    plt.title('Tap vs No Tap Over Frames')
    output_path = os.path.join(output_dir, 'tap_vs_frame.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot tap vs no tap from CSV")
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--out', type=str, default='airtap/outputs', help='Output folder')

    args = parser.parse_args()

    plot_csv(args.csv, args.out)