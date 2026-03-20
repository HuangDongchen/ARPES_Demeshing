# -*- coding: UTF-8 -*-
"""Command-line interface for ARPES demeshing.

Usage::

    arpes-demesh --image data/test.txt --num_iter 3000 --ckpt my_result
    arpes-demesh --image data/test.pxt --mode quality --mask_x 0 --mask_y 50 --mask_w 100 --mask_h 100
    arpes-demesh --image data/spectrum.ibw --num_iter 3000 --ckpt my_result
    arpes-demesh --image data/spectrum.ibw --output_fmt txt --ckpt my_result
"""
import argparse
import os


def main():
    """Entry point for the ``arpes-demesh`` console command."""
    parser = argparse.ArgumentParser(
        description='ARPES Demeshing — Dual U-Net Deep Image Prior',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O ---
    parser.add_argument("--image", type=str, required=True,
                        help="Input file path (.txt, .pxt, or .ibw)")
    parser.add_argument("--ckpt", type=str, default="result",
                        help="Output filename prefix")
    parser.add_argument("--output_dir", type=str, default="./checkpoint",
                        help="Output directory")
    parser.add_argument("--output_fmt", type=str, default=None,
                        choices=['txt', 'ibw'],
                        help="Output format (default: auto-detect from input)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device: 'auto' (detect best), 'cuda:0', 'mps', or 'cpu'")
    parser.add_argument("--row_cut_index", default=0, type=int,
                        help="Crop rows from bottom (0 = no crop)")

    # --- Algorithm ---
    parser.add_argument("--target_size", default=256, type=int,
                        help="LR processing resolution (square)")
    parser.add_argument("--mode", type=str, default='fast',
                        choices=['fast', 'quality'],
                        help="Upscale mode")
    parser.add_argument("--num_iter", default=2000, type=int,
                        help="Stage 1 iterations")
    parser.add_argument("--num_iter_fine", default=-1, type=int,
                        help="Stage 2 iterations (default: num_iter // 2)")
    parser.add_argument("--use_coord", action='store_true', default=True,
                        help="Coordinate injection for signal network")
    parser.add_argument("--no_coord", dest='use_coord', action='store_false',
                        help="Disable coordinate injection")
    parser.add_argument("--coord_mode", type=str, default='signal_only',
                        choices=['both', 'signal_only'])
    parser.add_argument("--mesh_l1", default=0.05, type=float,
                        help="L1 sparsity weight on mesh output")
    parser.add_argument("--width", default=128, type=int,
                        help="Network channel width")
    parser.add_argument("--num_scales", default=5, type=int,
                        help="U-Net downsampling scales")
    parser.add_argument("--skip_channels", default=4, type=int,
                        help="Skip connection channels")
    parser.add_argument("--lr_signal", default=0.5, type=float,
                        help="Signal network learning rate")
    parser.add_argument("--lr_mesh", default=1.0, type=float,
                        help="Mesh network learning rate")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=['sgd', 'adam'])
    parser.add_argument("--lambda_max", default=0.5, type=float,
                        help="Stage 2 regularization weight")

    # --- Mask ---
    parser.add_argument("--mask_x", type=int, default=None,
                        help="Mask rectangle top-left x (column)")
    parser.add_argument("--mask_y", type=int, default=None,
                        help="Mask rectangle top-left y (row)")
    parser.add_argument("--mask_w", type=int, default=None,
                        help="Mask rectangle width")
    parser.add_argument("--mask_h", type=int, default=None,
                        help="Mask rectangle height")

    args = parser.parse_args()


    from arpes_demeshing import demesh, load_ibw, load_pxt, load_txt, save_result, save_result_ibw
    from arpes_demeshing._mask import make_rect_mask

    # --- Load data ---
    image_path = args.image
    axes = None
    header = ''
    if image_path.endswith('.ibw'):
        data, axes = load_ibw(image_path)
    elif image_path.endswith('.pxt'):
        data, header = load_pxt(image_path)
    else:
        if not image_path.endswith('.txt'):
            image_path = image_path + '.txt'
        data, header = load_txt(image_path)

    # Determine output format
    output_fmt = args.output_fmt
    if output_fmt is None:
        output_fmt = 'ibw' if image_path.endswith('.ibw') else 'txt'

    if args.row_cut_index > 0:
        data = data[:args.row_cut_index, :]

    # --- Mask ---
    mask = None
    if args.mask_x is not None and args.mask_y is not None:
        mask = make_rect_mask(
            data.shape, args.mask_x, args.mask_y,
            args.mask_w or 100, args.mask_h or 100,
        )

    # --- Run ---
    result = demesh(
        data, mask=mask,
        target_size=args.target_size, mode=args.mode,
        num_iter=args.num_iter, num_iter_fine=args.num_iter_fine,
        use_coord=args.use_coord, coord_mode=args.coord_mode,
        mesh_l1=args.mesh_l1, width=args.width,
        num_scales=args.num_scales, skip_channels=args.skip_channels,
        lr_signal=args.lr_signal, lr_mesh=args.lr_mesh,
        optimizer=args.optimizer, lambda_max=args.lambda_max,
        device=args.device,
    )

    # --- Save ---
    if output_fmt == 'ibw':
        save_result_ibw(result, args.output_dir, args.ckpt, axes=axes, original=data)
    else:
        save_result(result, args.output_dir, args.ckpt, header, original=data)
    print(f"\nDone! Results saved to {args.output_dir}/{args.ckpt}_*")
    print(f"  Signal shape: {result.signal.shape}")
    print(f"  Signal range: [{result.signal.min():.1f}, {result.signal.max():.1f}]")


if __name__ == "__main__":
    main()
