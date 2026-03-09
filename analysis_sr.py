# -*- coding: UTF-8 -*-
# @Author  : Huang
# SR demeshing result analysis and visualization
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from arpes_demeshing.utils.denoising_utils import *  # noqa: F403


def plot(img, title: str, cmap=None):
    plt.imshow(img, aspect="auto", cmap='bwr')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()


def plot_texture(img, cnt: int, suffix=""):
    plt.imshow(img, aspect="auto", cmap='bwr')
    plt.title(f"Texture {cnt} {suffix}")
    plt.colorbar()
    plt.tight_layout()


parser = argparse.ArgumentParser(description='SR Demeshing Analysis')
parser.add_argument("--ckpt", type=str, default="sr_test", help="checkpoint name")
args = parser.parse_args()
print(args)

results = torch.load('./checkpoint/' + args.ckpt + '_last')

mode = results['mode']
original_shape = results['original_shape']
target_size = results['target_size']
print(f"Mode: {mode}, Original size: {original_shape}, Processing size: {target_size}")

# Extract components
texture_1_lr = np.squeeze(results['net1_out_lr'])
texture_2_lr = np.squeeze(results['net2_out_lr'])
texture_1_hr = np.squeeze(results['net1_out_hr'])
texture_2_hr = np.squeeze(results['net2_out_hr'])
ground_truth_hr = np.squeeze(results['ground_truth_hr'])

# ========== Low-resolution results ==========
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plot_texture(texture_1_lr, 1, "(LR)")
plt.subplot(1, 3, 2)
plot_texture(texture_2_lr, 2, "(LR)")
plt.subplot(1, 3, 3)
cor_lr = np.squeeze(results['ground_truth_lr']) - texture_1_lr - texture_2_lr
plot(cor_lr, "Residual (LR)", cmap='bwr')
plt.savefig("checkpoint/" + args.ckpt + "_lr_overview.png", dpi=150)
plt.close()

# ========== High-resolution results ==========
# Texture 1 HR (spectral signal)
plt.figure()
plot_texture(texture_1_hr, 1, f"(HR-{mode})")
plt.savefig("checkpoint/" + args.ckpt + "_texture1_hr.png")
np.savetxt("checkpoint/" + args.ckpt + "_texture_1_hr.txt", texture_1_hr, fmt="%.3f")

# Texture 2 HR (mesh artifact)
plt.figure()
plot_texture(texture_2_hr, 2, f"(HR-{mode})")
plt.savefig("checkpoint/" + args.ckpt + "_texture2_hr.png")
np.savetxt("checkpoint/" + args.ckpt + "_texture_2_hr.txt", texture_2_hr, fmt="%.3f")

# Residual: ground_truth - signal
residual = ground_truth_hr - texture_1_hr
plt.figure()
plot(residual, "Residual (HR)", cmap='bwr')
plt.savefig("checkpoint/" + args.ckpt + "_residual_hr.png")

# Original image
plt.figure()
plot(ground_truth_hr, "Original (HR)", cmap='bwr')
plt.savefig("checkpoint/" + args.ckpt + "_true_hr.png")
np.savetxt("checkpoint/" + args.ckpt + "_ground_truth_hr.txt", ground_truth_hr, fmt="%.3f")

# Gaussian-smoothed signal
texture_1_smoothed = cv2.GaussianBlur(texture_1_hr.astype(np.float32), (15, 15), 15)
plt.figure()
plot_texture(texture_1_smoothed, 1, "(HR smoothed)")
plt.savefig("checkpoint/" + args.ckpt + "_texture1_hr_smoothed.png")
np.savetxt("checkpoint/" + args.ckpt + "_texture_1_hr_smoothed.txt", texture_1_smoothed, fmt="%.3f")

# Loss history
plt.figure()
loss_1 = results['loss_history_stage1']
plt.plot(loss_1, label='Stage 1 (LR)')
if mode == 'quality' and results['loss_history_stage2']:
    loss_2 = results['loss_history_stage2']
    offset = len(loss_1)
    plt.plot(range(offset, offset + len(loss_2)), loss_2, label='Stage 2 (HR)')
plt.xlabel('Checkpoint Index')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig("checkpoint/" + args.ckpt + "_loss.png")
plt.close()

print(f"Analysis complete. Results saved to checkpoint/{args.ckpt}_*")
