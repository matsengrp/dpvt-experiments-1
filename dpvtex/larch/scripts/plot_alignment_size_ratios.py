import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_alignment_size_ratios(csv_path, output_path):
    """
    Plot the ratios of cleaned to original alignment sizes.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing alignment size statistics
    output_path : str
        Path to save the output plot
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(df['seq_ratio'], df['site_ratio'], alpha=0.6, s=50)

    # Add labels and title
    ax.set_xlabel('Sequence Ratio (cleaned/original)', fontsize=12)
    ax.set_ylabel('Site Ratio (cleaned/original)', fontsize=12)
    ax.set_title('Alignment Size Reduction After Cleaning', fontsize=14)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Set axis limits from 0 to 1
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # Add a diagonal line for reference (where seq_ratio == site_ratio)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='Equal reduction')
    ax.legend()

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def plot_cleaned_alignment_sizes(csv_path, output_path):
    """
    Plot cleaned alignment sizes colored by compression ratio.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing alignment size statistics
    output_path : str
        Path to save the output plot
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Calculate the overall compression ratio (product of seq_ratio and site_ratio)
    df['compression_ratio'] = df['seq_ratio'] * df['site_ratio']

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        df['cleaned_num_seqs'],
        df['cleaned_num_sites'],
        c=df['compression_ratio'],
        cmap='viridis',
        alpha=0.7,
        s=50,
        vmin=0,
        vmax=1
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Compression Ratio (seq_ratio × site_ratio)', fontsize=11)

    # Add labels and title
    ax.set_xlabel('Number of Sequences (cleaned)', fontsize=12)
    ax.set_ylabel('Number of Sites (cleaned)', fontsize=12)
    ax.set_title('Cleaned Alignment Sizes\n(colored by compression ratio)', fontsize=14)

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Add grid for better readability
    ax.grid(True, alpha=0.3, which='both')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
