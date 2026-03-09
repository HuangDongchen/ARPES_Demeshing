# -*- coding: UTF-8 -*-
# @Author  : Huang
# 2D demeshing result analysis and visualization
import argparse

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

from arpes_demeshing.utils.denoising_utils import *  # noqa: F403


def plot(img, title: str, cmap=None):
    plt.imshow(img, aspect="auto", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()


def plot_texture(img, cnt: int):
    plt.imshow(img, aspect="auto", cmap='bwr')
    plt.title("Texture %d" % (cnt))
    plt.colorbar()
    plt.tight_layout()


def plot_residual(img):
    plot(img, "Residual", cmap='bwr')


parser = argparse.ArgumentParser(description='2D Demeshing Analysis')
parser.add_argument("--ckpt", type=str, default="test", help="checkpoint name")

args = parser.parse_args()
print(args)

results = torch.load('./checkpoint/' + args.ckpt + '_last')

# Extract signal and mesh components
texture_1 = np.squeeze(results['net1_out'])
texture_2 = np.squeeze(results['net2_out'])
ground_truth = np.squeeze(results['ground_truth'])
cor = ground_truth - texture_1 - texture_2

# Plot and save texture 1 (signal)
plt.figure()
plot_texture(texture_1, 1)
plt.savefig("checkpoint/" + args.ckpt + "_texture1.png")
np.savetxt("checkpoint/" + args.ckpt + "_texture_1.txt", texture_1, fmt="%.3f")

# Plot and save texture 2 (mesh)
plt.figure()
plot_texture(texture_2, 2)
plt.savefig("checkpoint/" + args.ckpt + "_texture2.png")
np.savetxt("checkpoint/" + args.ckpt + "_texture_2.txt", texture_2, fmt="%.3f")

# Correlation (residual from both networks)
plt.figure()
plot_texture(cor, 2)
plt.savefig("checkpoint/" + args.ckpt + "_cor.png")

# Residual: ground_truth - signal
residual = ground_truth - texture_1
plt.figure()
plot_residual(residual)
plt.savefig("checkpoint/" + args.ckpt + "_residual.png")

# Original image
plt.figure()
plot(ground_truth, "Original", cmap='bwr')
plt.savefig("checkpoint/" + args.ckpt + "_true.png")
np.savetxt("checkpoint/" + args.ckpt + "_ground_truth.txt", ground_truth, fmt="%.3f")

# Gaussian-smoothed texture 1
texture_1_smoothed = cv.GaussianBlur(texture_1, (15, 15), 15)
plt.figure()
plot_texture(texture_1_smoothed, 1)
plt.savefig("checkpoint/" + args.ckpt + "_texture1_smoothed.png")
np.savetxt("checkpoint/" + args.ckpt + "_texture_1_smoothed.txt", texture_1_smoothed, fmt="%.3f")

print(f"Analysis complete. Results saved to checkpoint/{args.ckpt}_*")
