# Changelog

## v0.3.0 (2026-03-20)

### Added
- **IBW file support**: `load_ibw()` reads Igor Binary Wave v5 files with full axis preservation
- **IBW output**: `save_result_ibw()` saves demeshing results as `.ibw` files
- **CLI `--output_fmt`**: choose `txt` or `ibw` output format (auto-detects from input)
- **`__main__.py`**: support `python -m arpes_demeshing` invocation
- **3D warning**: `load_ibw()` emits `UserWarning` for >2D data, slicing to first 2D plane

### Changed
- Extracted `_save_visualizations()` as shared helper for txt/ibw save paths
- `demesh()` gains `output_fmt` and `output_axes` parameters for IBW auto-save
- Package structure: `utils/ibw.py` added as internal IBW reader/writer

### Fixed
- Lint cleanup in `utils/ibw.py` (split imports, remove unused symbols, fix multi-statements)

## v0.2.0

- Coordinate injection (CoordConv) for signal network
- Mask support for excluding regions from loss
- PXT file reading via `igor2`
- L1 sparsity regularization on mesh network

## v0.1.0

- Initial release: dual U-Net Deep Image Prior demeshing
- Fast mode (Lanczos upscale) and quality mode (coarse-to-fine)
- TXT file I/O with header preservation
- CLI entry point `arpes-demesh`
