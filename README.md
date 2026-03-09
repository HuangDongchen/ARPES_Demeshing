# ARPES Demeshing: Dual U-Net Deep Image Prior for Mesh Artifact Removal

A PyTorch-based tool for removing mesh (grid) artifacts from Angle-Resolved Photoemission Spectroscopy (ARPES) data using a dual-network Deep Image Prior approach.

## Method Overview

This tool employs **two independent U-Net networks** to perform unsupervised signal–artifact separation:

- **Network 1 (Signal)**: Captures the smooth spectral features of the ARPES measurement
- **Network 2 (Mesh)**: Captures the periodic mesh/grid artifact pattern

The total reconstruction equals the sum of both networks' outputs. By exploiting the different spectral priors learned by each network, the method separates the clean signal from the mesh pattern without any supervised training data.

### Key Features

- **Super-Resolution Pipeline**: Downscale → demesh at low resolution → upscale, with two modes:
  - `fast`: Lanczos upscale (default, quick)
  - `quality`: Coarse-to-fine refinement (higher quality, slower)
- **Mask Support**: Exclude specific regions (e.g. bright Fermi surface) from loss computation
- **Asymmetric Coordinate Injection**: Normalized (x, y) coordinates for improved spatial awareness
- **L1 Sparsity Regularization**: Encourages the mesh network to produce sparse outputs
- **PXT File Support**: Direct reading of Igor Pro `.pxt` packed experiment files
- **Auto-Save**: One-call `demesh(save=True)` generates data files and comparison plots

## Installation

```bash
# Clone and install
git clone https://github.com/huangdongchen/arpes_demeshing.git
cd arpes_demeshing
pip install -e .

# For Igor Pro .pxt file support
pip install igor2
```

**Requirements**: Python 3.7+, CUDA-capable GPU recommended.

## Quick Start

### Python API (Recommended)

```python
import arpes_demeshing as ad

# Load data
data, header = ad.load_txt("data/your_spectrum.txt")  # or ad.load_pxt("data/file.pxt")

# Run demeshing with auto-save
result = ad.demesh(
    data,
    target_size=256,       # processing resolution
    mode='fast',           # 'fast' or 'quality'
    num_iter=4000,         # training iterations
    save=True,             # auto-save results + comparison plot
    output_dir='./output', # where to save
    output_name='my_run',  # filename prefix
)

# Access results directly
clean_signal = result.signal   # (H, W) numpy array — clean spectrum
mesh_pattern = result.mesh     # (H, W) numpy array — extracted mesh
```

This will automatically generate:
- `my_run_signal.txt` / `my_run_mesh.txt` — numerical data
- `my_run_signal.png` / `my_run_mesh.png` — individual visualizations
- `my_run_comparison.png` — 4-panel comparison (Original / Signal / Mesh / Residual)
- `my_run_loss.png` — training loss curve

### With Mask (exclude regions from fitting)

```python
import arpes_demeshing as ad

data, header = ad.load_txt("data/your_spectrum.txt")

# Create a rectangular mask — pixels inside are excluded from loss
mask = ad.make_rect_mask(data.shape, x=0, y=50, w=100, h=100)

result = ad.demesh(data, mask=mask, save=True, output_dir='./output', output_name='masked_run')
```

### Command-Line Interface

```bash
# Basic usage
arpes-demesh --image data/your_data.txt --num_iter 4000 --ckpt my_result

# With PXT file and mask
arpes-demesh --image data/file.pxt --mask_x 0 --mask_y 50 --mask_w 100 --mask_h 100

# Quality mode (coarse-to-fine refinement)
arpes-demesh --image data/your_data.txt --mode quality --num_iter 4000
```

## Complete Example

```python
import arpes_demeshing as ad
import matplotlib.pyplot as plt

# Step 1: Load ARPES data
data, header = ad.load_txt("data/ATest20250910120632.txt")
print(f"Loaded: {data.shape}, range=[{data.min():.1f}, {data.max():.1f}]")

# Step 2: Run demeshing (with auto-save)
result = ad.demesh(
    data,
    target_size=256,
    mode='fast',
    num_iter=3000,
    mesh_l1=0.05,
    save=True,
    output_dir='./checkpoint',
    output_name='example',
)

# Step 3: Inspect results
print(f"Signal shape: {result.signal.shape}")
print(f"Final loss: {result.loss_history[-1]:.6f}")

# Step 4: Further analysis on the clean signal
# result.signal is a standard numpy array — use it with any analysis tool
from scipy.ndimage import gaussian_filter
smoothed = gaussian_filter(result.signal, sigma=2)

# Step 5: Or generate a custom comparison plot
fig = ad.plot_comparison(data, result)
plt.show()
```

## Package Structure

```
arpes_demeshing/
├── pyproject.toml               # pip install configuration
├── README.md
├── LICENSE
├── arpes_demeshing/             # Python package
│   ├── __init__.py              # Public API
│   ├── core.py                  # demesh() — main algorithm
│   ├── _config.py               # DemeshConfig / DemeshResult
│   ├── _mask.py                 # Mask utilities
│   ├── io.py                    # load_txt / load_pxt / save_result / plot_comparison
│   ├── cli.py                   # arpes-demesh CLI entry point
│   ├── models/                  # Neural network architectures (U-Net)
│   └── utils/                   # Image/tensor conversion, scaling, PXT reader
├── data/                        # Place input files here
└── checkpoint/                  # Default output directory
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_size` | 256 | Processing resolution for Stage 1 (square) |
| `mode` | `fast` | `fast` (Lanczos upscale) or `quality` (coarse-to-fine) |
| `num_iter` | 2000 | Stage 1 training iterations |
| `num_iter_fine` | auto | Stage 2 iterations (default: `num_iter // 2`) |
| `mask` | `None` | Boolean mask — `True` = excluded from loss |
| `mesh_l1` | 0.05 | L1 sparsity weight on mesh network output |
| `width` | 128 | Network channel width |
| `use_coord` | `True` | Append (x, y) coordinate channels to signal network |
| `save` | `False` | Auto-save results + comparison plot |
| `device` | `cuda:0` | Torch device (`cuda:0`, `cpu`, etc.) |

## How It Works

1. **Input**: Raw ARPES spectrum with mesh artifacts (2D intensity matrix)
2. **Preprocessing**: Normalize to [0, 1]; downscale for efficiency
3. **Dual-Network Optimization**: Two U-Nets jointly minimize reconstruction loss:
   - `loss = MSE(net1 + net2, target) + λ · |net2|₁`
4. **Separation**: Network 1 output ≈ clean signal; Network 2 output ≈ mesh pattern
5. **Post-processing**: Upscale to original resolution, denormalize, optionally save

The separation works because:
- The **signal network** (deeper, with coordinate input) naturally captures smooth, spatially-varying spectral features
- The **mesh network** (shallower, without coordinates) captures repetitive grid patterns
- **L1 regularization** on the mesh output encourages sparsity, preventing signal leakage

## Citation

If you find this tool useful in your research, please cite:

```
@article{PhysRevB.107.165106,
  title = {Removing grid structure in angle-resolved photoemission spectra via deep learning method},
  author = {Liu, Junde and Huang, Dongchen and Yang, Yi-feng and Qian, Tian},
  journal = {Phys. Rev. B},
  volume = {107},
  issue = {16},
  pages = {165106},
  numpages = {9},
  year = {2023},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.107.165106},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.107.165106}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
