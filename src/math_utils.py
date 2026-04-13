import numpy as np
import torch.fft as tfft
import torch
import math
from typing import Tuple, List
from torch.nn import functional as F
from scipy.ndimage import uniform_filter
from torchmetrics.functional import structural_similarity_index_measure as ssim


# NumPy Surface Coil Correction (SCC) function
def compute_scc(
        pd_image, filter_size=7,
        thres_ratio_for_noise=4.0, noise_background=1.0
) -> np.ndarray:
    """
    Compute the Surface Coil Correction (SCC) map.

    Parameters:
    - PD: 2D complex-valued NumPy array representing the proton density image.
    - filter_size: Size of the square box filter (must be odd).
    - thres_ratio_for_noise: Threshold multiplier for regularization.
    - noise_background: Standard deviation of noise.

    Returns:
    - scc: 2D real-valued NumPy array representing the SCC map.
    """
    # Ensure filter_size is odd
    if filter_size % 2 == 0:
        filter_size += 1

    pd_mag = np.abs(pd_image)
    pd_mag_sq = pd_mag ** 2

    # Apply box filtering using uniform_filter with circular padding
    pd_filtered = uniform_filter(pd_mag, size=filter_size, mode='wrap')
    pd_sq_filtered = uniform_filter(pd_mag_sq, size=filter_size, mode='wrap')

    # Compute regularization term
    regularization = (noise_background * thres_ratio_for_noise) ** 2
    return pd_filtered / (pd_sq_filtered + regularization)


# Batch processing for NumPy SCC
def compute_scc_batch(
        pd_image, filter_size=7,
        thres_ratio_for_noise=4.0, noise_background=1.0
) -> np.ndarray:
    return np.stack([
        compute_scc(im, filter_size, thres_ratio_for_noise, noise_background)
        for im in pd_image
    ])


# NumPy centered FFT functions
def fftc(
    x: np.ndarray,
    axes: Tuple[int, int] = (-2, -1)
):
    x_shifted = np.fft.ifftshift(x, axes=axes)
    k = np.fft.fft2(x_shifted, axes=axes, norm='ortho')
    k_centered = np.fft.fftshift(k, axes=axes)
    return k_centered


def ifftc(
    k: np.ndarray,
    axes: Tuple[int, int] = (-2, -1)
):
    k_shifted = np.fft.ifftshift(k, axes=axes)
    x = np.fft.ifft2(k_shifted, axes=axes, norm='ortho')
    x_centered = np.fft.fftshift(x, axes=axes)
    return x_centered


@torch.jit.script
def norm_tensor(
    x: torch.Tensor,
    axes: Tuple[int, int] = (-2, -1)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize a 4D complex valued tensor along
    the last two dims, i.e., the spatial dims
    """
    mu = torch.mean(x, dim=axes, keepdim=True)
    sig = torch.std(x, dim=axes, keepdim=True)
    x = (x - mu) / sig
    return x, mu, sig


@torch.jit.script
def unnorm_tensor(
    x: torch.Tensor,
    mu: torch.Tensor,
    sig: torch.Tensor
) -> torch.Tensor:
    """
    Unnormalize a 4D complex valued tensor along
    the last two dims, i.e., the spatial dims
    mu and sig should be computed using norm_tensor
    """
    return x * sig + mu


@torch.jit.script
# Torch centered FFT functions
def tfftc(
    x: torch.Tensor,
    axes: Tuple[int, int] = (-2, -1)
) -> torch.Tensor:
    """Centered FFT"""
    x_shifted = tfft.ifftshift(x, dim=axes)
    k = tfft.fft2(x_shifted, dim=axes, norm='ortho')
    k_centered = tfft.fftshift(k, dim=axes)
    return k_centered


@torch.jit.script
def itfftc(
    k: torch.Tensor,
    axes: Tuple[int, int] = (-2, -1)
) -> torch.Tensor:
    """Inverse centered FFT"""
    k_shifted = tfft.ifftshift(k, dim=axes)
    x = tfft.ifft2(k_shifted, dim=axes, norm='ortho')
    x_centered = tfft.fftshift(x, dim=axes)
    return x_centered


@torch.jit.script
def expand(x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """ Expand coil combined image to multi-coil image"""
    return sens_maps * x


@torch.jit.script
def reduce(xc: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
    """ Reduce multi-coil image to match filter combined image"""
    return torch.sum(torch.conj(sens_maps) * xc, dim=1, keepdim=True)


@torch.jit.script
def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to channel dimension for real-valued networks."""
    x = torch.view_as_real(x)
    b, c, h, w, two = x.shape
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


@torch.jit.script
def chan_dim_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert channel dimension back to complex tensor."""
    b, c2, h, w = x.shape
    c = c2 // 2
    out = x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    return torch.view_as_complex(out)


@torch.jit.script
def pad(
    x: torch.Tensor
) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
    _, _, h, w = x.shape
    w_mult = ((w - 1) | 15) + 1
    h_mult = ((h - 1) | 15) + 1
    w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
    h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
    x = F.pad(x, w_pad + h_pad)
    return x, (h_pad, w_pad, h_mult, w_mult)


@torch.jit.script
def unpad(
    x: torch.Tensor,
    h_pad: List[int],
    w_pad: List[int],
    h_mult: int,
    w_mult: int,
) -> torch.Tensor:
    return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]


@torch.jit.script
def per_slice_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize each slice in the batch to [0, 1] range.
    Assumes x is of shape (B, 1, H, W)
    eps is used to avoid division by zero.
    """
    batch_size = x.shape[0]
    x_flat = x.view(batch_size, -1)
    min_val = x_flat.min(dim=1, keepdim=True)[0]
    min_val = min_val.view(-1, 1, 1, 1)
    max_val = x_flat.max(dim=1, keepdim=True)[0]
    max_val = max_val.view(-1, 1, 1, 1)
    return (x - min_val) / torch.clamp(max_val - min_val, min=eps)


@torch.jit.script
def compute_scc_torch(
    pd_image: torch.Tensor,
    filter_size: int = 7,
    thres_ratio_for_noise: float = 4.0,
    noise_background: float = 1.0
) -> torch.Tensor:
    """
    Compute surface coil correction for a given MOCO Avg PD image.
    """
    device = pd_image.device
    pd_mag = torch.abs(pd_image)
    pd_mag_sq = pd_mag ** 2
    kernel = torch.full(
        (1, 1, filter_size, filter_size),
        1.0 / (filter_size ** 2),
        device=device,
        dtype=pd_mag.dtype
    )
    pad_size = filter_size // 2
    pad_tuple = (pad_size, pad_size, pad_size, pad_size)
    pd_mag_padded = F.pad(pd_mag, pad_tuple, mode='circular')
    pd_mag_sq_padded = F.pad(pd_mag_sq, pad_tuple, mode='circular')
    pd_filtered = F.conv2d(pd_mag_padded, kernel, padding=0)
    pd_sq_filtered = F.conv2d(pd_mag_sq_padded, kernel, padding=0)
    regularization = (noise_background * thres_ratio_for_noise) ** 2
    return pd_filtered / (pd_sq_filtered + regularization)


def batch_ssim(
        pred, target, min_target, max_target, eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes the average SSIM for a batch of reconstructions and targets.
    Targets do not need to be normalized to [0, 1] range.
    """
    B = pred.shape[0]
    # Ensure min_target and max_target have correct shape for broadcasting
    min_target = min_target.view(B, 1, 1, 1)
    max_target = max_target.view(B, 1, 1, 1)

    # Compute data range per slice
    data_ranges = torch.clamp(max_target - min_target, min=eps)

    # Normalize each slice to [0, 1] range using broadcasting
    pred_normalized = (pred - min_target) / data_ranges
    target_normalized = (target - min_target) / data_ranges

    # Calculate SSIM - inputs are already (B, 1, H, W)
    return ssim(
        pred_normalized, target_normalized,
        gaussian_kernel=False,
        kernel_size=7,
        data_range=(0.0, 1.0)
    )  # type: ignore
