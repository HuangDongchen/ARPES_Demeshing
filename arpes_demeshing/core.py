# -*- coding: UTF-8 -*-
"""Core demeshing algorithm — Dual U-Net Deep Image Prior.

Provides the main ``demesh()`` function that separates ARPES signal from
mesh artifacts using two independent U-Net networks.
"""
import numpy as np
import torch

from ._config import DemeshConfig, DemeshResult
from ._mask import masked_mse_loss
from .models.skip import skip
from .utils.common_utils import (
    crop_array,
    get_noise,
    get_params,
    image_normalization,
    np_to_torch,
    torch_to_np,
)
from .utils.sr_utils import downscale, upscale

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_coord_input(h, w):
    """Generate normalized coordinate channels (1, 2, H, W), range [-1, 1]."""
    y = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(1, 1, h, w)
    x = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(1, 1, h, w)
    return torch.cat([x, y], dim=1)


def _build_networks(h, w, config, device, input_depth=32):
    """Build dual U-Net networks for the given image size."""
    noise_depth = input_depth
    net1_depth = input_depth + 2 if config.use_coord else input_depth
    net2_depth = input_depth + 2 if (config.use_coord and config.coord_mode == 'both') else input_depth

    n_channels = 1
    num_scales = config.num_scales
    num_scales_2 = max(3, num_scales - 2)
    skip_n11 = config.skip_channels

    # net_1: signal network (deeper, with optional coord input)
    net_1 = skip(
        net1_depth, n_channels,
        num_channels_down=[config.width] * num_scales,
        num_channels_up=[config.width] * num_scales,
        num_channels_skip=[skip_n11] * num_scales,
        upsample_mode='bilinear', downsample_mode='stride',
        filter_size_down=5, filter_size_up=5,
        need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU',
    ).to(device)

    # net_2: mesh network (shallower)
    net_2 = skip(
        net2_depth, n_channels,
        num_channels_down=[config.width] * num_scales_2,
        num_channels_up=[config.width] * num_scales_2,
        num_channels_skip=[skip_n11] * num_scales_2,
        upsample_mode='bilinear', downsample_mode='stride',
        filter_size_down=5, filter_size_up=5,
        need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU',
    ).to(device)

    net_1_input = get_noise(noise_depth, 'noise', (h, w)).to(device).detach()
    net_2_input = get_noise(noise_depth, 'noise', (h, w)).to(device).detach()

    if config.use_coord:
        coord = _make_coord_input(h, w).to(device)
        net_1_input = torch.cat([net_1_input, coord], dim=1)
        if config.coord_mode == 'both':
            net_2_input = torch.cat([net_2_input, coord.clone()], dim=1)

    return net_1, net_2, net_1_input, net_2_input, noise_depth


def _build_optimizers(net_1, net_2, net_1_input, net_2_input, config):
    """Build optimizers for both networks."""
    p_1 = get_params('net', net_1, net_1_input)
    p_2 = get_params('net', net_2, net_2_input)
    if config.optimizer == 'adam':
        return torch.optim.Adam(p_1, lr=config.lr_signal), torch.optim.Adam(p_2, lr=config.lr_mesh)
    elif config.optimizer == 'sgd':
        return torch.optim.SGD(p_1, lr=config.lr_signal), torch.optim.SGD(p_2, lr=config.lr_mesh)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def _perturb_inputs(net_1_input, net_2_input, config, device, reg_noise_std):
    """Add Gaussian perturbation to noise channels, keeping coord channels intact."""
    if config.use_coord:
        n1_ch = net_1_input.shape[1] - 2
        noise_1 = torch.zeros(1, n1_ch, net_1_input.shape[2], net_1_input.shape[3],
                               device=device).normal_(std=reg_noise_std)
        pad_1 = torch.zeros(1, 2, net_1_input.shape[2], net_1_input.shape[3], device=device)
        inp1 = net_1_input + torch.cat([noise_1, pad_1], dim=1)

        if config.coord_mode == 'both':
            n2_ch = net_2_input.shape[1] - 2
            noise_2 = torch.zeros(1, n2_ch, net_2_input.shape[2], net_2_input.shape[3],
                                   device=device).normal_(std=reg_noise_std)
            pad_2 = torch.zeros(1, 2, net_2_input.shape[2], net_2_input.shape[3], device=device)
            inp2 = net_2_input + torch.cat([noise_2, pad_2], dim=1)
        else:
            inp2 = net_2_input + torch.zeros_like(net_2_input).normal_(std=reg_noise_std)
    else:
        inp1 = net_1_input + torch.zeros_like(net_1_input).normal_(std=reg_noise_std)
        inp2 = net_2_input + torch.zeros_like(net_2_input).normal_(std=reg_noise_std)
    return inp1, inp2


def _train_stage(net_1, net_2, net_1_input, net_2_input,
                 target_torch, mask_torch, num_iter, config, device,
                 stage_name="", verbose=True, callback=None):
    """Train one stage (no regularization constraint)."""
    criterion = torch.nn.MSELoss().to(device)
    reg_noise_std = 1. / 30.
    show_every = 100
    loss_history = []
    opt_1, opt_2 = _build_optimizers(net_1, net_2, net_1_input, net_2_input, config)

    for it in range(num_iter):
        opt_1.zero_grad()
        opt_2.zero_grad()

        inp1, inp2 = _perturb_inputs(net_1_input, net_2_input, config, device, reg_noise_std)
        out1 = net_1(inp1)
        out2 = net_2(inp2)
        r_img = out1 + out2

        if mask_torch is not None:
            mse = masked_mse_loss(r_img, target_torch, mask_torch)
        else:
            mse = criterion(r_img, target_torch)

        total_loss = mse + config.mesh_l1 * out2.abs().mean() if config.mesh_l1 > 0 else mse
        total_loss.backward()
        opt_1.step()
        opt_2.step()

        if it % show_every == 0 or it == num_iter - 1:
            if verbose:
                print(f'[{stage_name}] Iter {it:05d}  Loss {total_loss.item():.6f}')
            loss_history.append(total_loss.item())
            if callback:
                callback(it, total_loss.item(), torch_to_np(out1), torch_to_np(out2))

    with torch.no_grad():
        out1 = net_1(net_1_input)
        out2 = net_2(net_2_input)
    return loss_history, torch_to_np(out1 + out2), torch_to_np(out1), torch_to_np(out2)


def _train_stage_c2f(net_1, net_2, net_1_input, net_2_input,
                     target_torch, mask_torch, num_iter,
                     ref_t1_torch, ref_t2_torch, lambda_max,
                     config, device, stage_name="", verbose=True, callback=None):
    """Coarse-to-fine training with regularization annealing."""
    criterion = torch.nn.MSELoss().to(device)
    reg_noise_std = 1. / 30.
    show_every = 100
    loss_history = []
    opt_1, opt_2 = _build_optimizers(net_1, net_2, net_1_input, net_2_input, config)

    for it in range(num_iter):
        opt_1.zero_grad()
        opt_2.zero_grad()

        lam = lambda_max * (1.0 - it / num_iter)

        inp1, inp2 = _perturb_inputs(net_1_input, net_2_input, config, device, reg_noise_std)
        out1 = net_1(inp1)
        out2 = net_2(inp2)
        r_img = out1 + out2

        if mask_torch is not None:
            main_loss = masked_mse_loss(r_img, target_torch, mask_torch)
        else:
            main_loss = criterion(r_img, target_torch)

        reg_loss = criterion(out1, ref_t1_torch) + criterion(out2, ref_t2_torch)
        l1_term = config.mesh_l1 * out2.abs().mean() if config.mesh_l1 > 0 else 0.0
        total_loss = main_loss + lam * reg_loss + l1_term
        total_loss.backward()
        opt_1.step()
        opt_2.step()

        if it % show_every == 0 or it == num_iter - 1:
            if verbose:
                print(f'[{stage_name}] Iter {it:05d}  Loss {total_loss.item():.6f}'
                      f'  main={main_loss.item():.6f} reg={reg_loss.item():.6f} lam={lam:.4f}')
            loss_history.append(total_loss.item())
            if callback:
                callback(it, total_loss.item(), torch_to_np(out1), torch_to_np(out2))

    with torch.no_grad():
        out1 = net_1(net_1_input)
        out2 = net_2(net_2_input)
    return loss_history, torch_to_np(out1 + out2), torch_to_np(out1), torch_to_np(out2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def demesh(
    data,
    mask=None,
    target_size=256,
    mode='fast',
    num_iter=2000,
    num_iter_fine=-1,
    use_coord=True,
    coord_mode='signal_only',
    mesh_l1=0.05,
    width=128,
    num_scales=5,
    skip_channels=4,
    lr_signal=0.5,
    lr_mesh=1.0,
    optimizer='sgd',
    lambda_max=0.5,
    device='cuda:0',
    verbose=True,
    callback=None,
    save=False,
    output_dir='./checkpoint',
    output_name='result',
):
    """Remove mesh artifacts from a 2D ARPES spectrum.

    Args:
        data: 2D numpy array (H, W) — raw ARPES intensity matrix.
        mask: Optional 2D numpy array (H, W), float32 or bool.
              1.0 / True = excluded from loss (e.g. bright Fermi region).
        target_size: Processing resolution for Stage 1 (square).
        mode: ``'fast'`` (Lanczos upscale) or ``'quality'`` (coarse-to-fine HR).
        num_iter: Stage 1 training iterations.
        num_iter_fine: Stage 2 iterations (default: ``num_iter // 2``).
        use_coord: Append (x, y) coordinate channels to signal network.
        coord_mode: ``'signal_only'`` or ``'both'`` (which networks get coords).
        mesh_l1: L1 sparsity weight on mesh network output.
        width: Network channel width.
        num_scales: U-Net downsampling scales for signal network.
        skip_channels: Skip connection channels.
        lr_signal: Learning rate for signal network.
        lr_mesh: Learning rate for mesh network.
        optimizer: ``'sgd'`` or ``'adam'``.
        lambda_max: Stage 2 regularization weight (annealed to 0).
        device: Torch device string, e.g. ``'cuda:0'`` or ``'cpu'``.
        verbose: Print progress messages.
        callback: Optional ``callback(step, loss, signal_np, mesh_np)``.
        save: If True, auto-save results (txt + png + comparison) to *output_dir*.
        output_dir: Directory for saved files (used when ``save=True``).
        output_name: Filename prefix for saved files.

    Returns:
        DemeshResult: Contains ``signal``, ``mesh``, ``loss_history``, etc.
    """
    if num_iter_fine < 0:
        num_iter_fine = num_iter // 2

    config = DemeshConfig(
        target_size=target_size, mode=mode, num_iter=num_iter,
        num_iter_fine=num_iter_fine, use_coord=use_coord,
        coord_mode=coord_mode, mesh_l1=mesh_l1, width=width,
        num_scales=num_scales, skip_channels=skip_channels,
        lr_signal=lr_signal, lr_mesh=lr_mesh, optimizer=optimizer,
        lambda_max=lambda_max,
    )
    dev = torch.device(device)

    # --- Preprocessing ---
    img_raw = crop_array(data, d=1)
    original_shape = img_raw.shape
    img_raw_np, norm_const = image_normalization(img_raw[np.newaxis, :, :])

    # Downscale
    img_lr = downscale(np.squeeze(img_raw_np), (target_size, target_size))
    img_torch = np_to_torch(img_lr[np.newaxis, :, :]).float().to(dev)

    # Mask handling
    mask_lr_torch = None
    mask_hr_torch = None
    if mask is not None:
        mask_cropped = crop_array(mask.astype(np.float32), d=1)
        mask_lr = downscale(mask_cropped, (target_size, target_size))
        mask_lr = (mask_lr > 0.5).astype(np.float32)
        mask_lr_torch = np_to_torch(mask_lr[np.newaxis, :, :]).float().to(dev)
        mask_hr_torch = np_to_torch(mask_cropped[np.newaxis, :, :]).float().to(dev)

    # --- Stage 1: LR demeshing ---
    if verbose:
        print(f"\nStage 1: LR demeshing ({target_size}x{target_size})")

    net_1, net_2, net_1_input, net_2_input, _ = _build_networks(
        target_size, target_size, config, dev,
    )
    loss_1, _, net1_out_lr, net2_out_lr = _train_stage(
        net_1, net_2, net_1_input, net_2_input,
        img_torch, mask_lr_torch, num_iter, config, dev,
        stage_name="Stage1-LR", verbose=verbose, callback=callback,
    )

    # --- Stage 2: upscale ---
    if mode == 'fast':
        if verbose:
            print("Fast mode: Lanczos upscale")
        texture_1_hr = upscale(np.squeeze(net1_out_lr), original_shape)
        texture_2_hr = upscale(np.squeeze(net2_out_lr), original_shape)
        loss_2 = []

    elif mode == 'quality':
        ref_t1_np = upscale(np.squeeze(net1_out_lr), original_shape)
        ref_t2_np = upscale(np.squeeze(net2_out_lr), original_shape)
        ref_t1_torch = np_to_torch(ref_t1_np[np.newaxis, :, :]).float().to(dev)
        ref_t2_torch = np_to_torch(ref_t2_np[np.newaxis, :, :]).float().to(dev)

        c2f_depth = 34  # 32 noise + 2 reference channels
        if verbose:
            print(f"Stage 2: HR refinement ({original_shape[0]}x{original_shape[1]}, {num_iter_fine} iter)")

        net_1_hr, net_2_hr, inp_hr_1, inp_hr_2, _ = _build_networks(
            original_shape[0], original_shape[1], config, dev, input_depth=c2f_depth,
        )
        # Inject reference textures into last 2 channels
        inp_hr_1[:, 32:33, :, :] = ref_t1_torch
        inp_hr_1[:, 33:34, :, :] = ref_t2_torch
        inp_hr_2[:, 32:33, :, :] = ref_t1_torch
        inp_hr_2[:, 33:34, :, :] = ref_t2_torch

        img_hr_torch = np_to_torch(img_raw_np).float().to(dev)

        loss_2, _, net1_out_hr, net2_out_hr = _train_stage_c2f(
            net_1_hr, net_2_hr, inp_hr_1, inp_hr_2,
            img_hr_torch, mask_hr_torch, num_iter_fine,
            ref_t1_torch, ref_t2_torch, lambda_max,
            config, dev, stage_name="Stage2-C2F", verbose=verbose, callback=callback,
        )
        texture_1_hr = np.squeeze(net1_out_hr)
        texture_2_hr = np.squeeze(net2_out_hr)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Build result ---
    result = DemeshResult(
        signal=texture_1_hr * norm_const,
        mesh=texture_2_hr * norm_const,
        signal_norm=texture_1_hr,
        mesh_norm=texture_2_hr,
        loss_history=loss_1 + loss_2,
        config=config,
        norm_const=norm_const,
    )

    # --- Auto-save ---
    if save:
        from .io import save_result as _save_result
        _save_result(result, output_dir, output_name, original=data)
        if verbose:
            print(f"Results saved to {output_dir}/{output_name}_*")

    return result
