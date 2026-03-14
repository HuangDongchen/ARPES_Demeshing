# -*- coding: UTF-8 -*-
"""Data loading and result saving utilities."""
import os

import matplotlib.pyplot as plt
import numpy as np


def load_txt(path):
    """Load ARPES data from a tab-separated txt file.

    Expected format: first line = header string, remaining lines = numeric data.
    Also supports SES format files where data starts after a `[Data 1]` marker.

    Args:
        path: Path to the .txt file.

    Returns:
        (data, header): 2D numpy array and the header string.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    header_lines = []
    data_start_idx = 0
    is_ses_format = False

    for i, line in enumerate(lines):
        line_strip = line.strip()
        if line_strip == '[Data 1]':
            header_lines.append(line.rstrip('\n'))
            data_start_idx = i + 1
            is_ses_format = True
            break
        header_lines.append(line.rstrip('\n'))

    if not is_ses_format:
        # Fallback to standard 1-line header format
        header_lines = [lines[0].rstrip('\n')]
        data_start_idx = 1

    header = '\n'.join(header_lines)
    # Load the remaining lines into a numeric array
    data = np.loadtxt(lines[data_start_idx:])
    return data, header


def load_pxt(path):
    """Load ARPES data from an Igor Pro .pxt packed experiment file.

    Requires the ``igor2`` package (``pip install igor2``).

    Args:
        path: Path to the .pxt file.

    Returns:
        (data, wave_name): 2D numpy array and the Igor wave name.
    """
    from .utils.pxt_utils import load_pxt as _load_pxt
    return _load_pxt(path)


def plot_comparison(original, result, output_path=None, dpi=150):
    """Generate a 4-panel comparison figure.

    Layout::

        [Original]          [Signal (cleaned)]
        [Mesh artifact]     [Residual]

    Args:
        original: Raw input data (H, W) numpy array (denormalized).
        result: A ``DemeshResult`` instance.
        output_path: If given, save the figure to this path.
        dpi: Figure resolution.

    Returns:
        matplotlib.figure.Figure
    """
    residual = original - result.signal - result.mesh

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Original
    im0 = axes[0, 0].imshow(original, aspect='auto', cmap='bwr')
    axes[0, 0].set_title('Original')
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Signal
    im1 = axes[0, 1].imshow(result.signal, aspect='auto', cmap='bwr')
    axes[0, 1].set_title('Signal (cleaned)')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Mesh
    im2 = axes[1, 0].imshow(result.mesh, aspect='auto', cmap='bwr')
    axes[1, 0].set_title('Mesh artifact')
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # Residual
    im3 = axes[1, 1].imshow(residual, aspect='auto', cmap='bwr')
    axes[1, 1].set_title('Residual')
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    return fig


def save_result(result, output_dir, name, header='', original=None):
    """Save a DemeshResult to disk (txt + png + comparison).

    Outputs:
        - ``<name>_signal.txt`` — denormalized clean signal
        - ``<name>_mesh.txt`` — denormalized mesh pattern
        - ``<name>_signal.png`` / ``<name>_mesh.png`` — visualizations
        - ``<name>_loss.png`` — training loss curve
        - ``<name>_comparison.png`` — 4-panel comparison (if *original* given)

    Args:
        result: A ``DemeshResult`` instance.
        output_dir: Directory for output files (created if needed).
        name: Filename prefix.
        header: Optional header line written to the txt file.
        original: Optional raw input array for comparison plot.

    Returns:
        str: Path to the saved signal txt file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- txt files ---
    txt_path = os.path.join(output_dir, name + '_signal.txt')
    with open(txt_path, 'w') as f:
        if header:
            f.write(header + '\n')
        np.savetxt(f, result.signal, fmt='%.6f', delimiter='\t')

    mesh_path = os.path.join(output_dir, name + '_mesh.txt')
    np.savetxt(mesh_path, result.mesh, fmt='%.6f', delimiter='\t')

    # --- individual visualizations ---
    plt.figure()
    plt.imshow(result.signal_norm, aspect='auto', cmap='bwr')
    plt.colorbar()
    plt.title('Signal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name + '_signal.png'), dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(result.mesh_norm, aspect='auto', cmap='bwr')
    plt.colorbar()
    plt.title('Mesh')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name + '_mesh.png'), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(result.loss_history)
    plt.xlabel('Checkpoint Index')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, name + '_loss.png'))
    plt.close()

    # --- comparison panel ---
    if original is not None:
        comp_path = os.path.join(output_dir, name + '_comparison.png')
        plot_comparison(original, result, output_path=comp_path)

    return txt_path

