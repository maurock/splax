"""Utility functions"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def project_depth_to_3d(
    depth_map: Float[Array, "H W"], fx: float, fy: float, cx: float, cy: float
) -> Float[Array, "H W 3"]:
    """Project depth map to 3D point cloud."""
    H, W = depth_map.shape
    i, j = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    z = depth_map
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    points_3d = jnp.stack((x, y, z), axis=-1)
    return points_3d


def transform_to_world(points, c2w_mat):
    # points: [H, W, 3], c2w_mat: [4, 4]
    R = c2w_mat[:3, :3]
    T = c2w_mat[:3, 3]
    return points @ R.T + T


def ssim(
    img1: Float[Array, "H W C"],
    img2: Float[Array, "H W C"],
    *,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Float[Array, ""]:
    """Computes the structural similarity index (SSIM) between an image pair.

    Args:
        img1: First image.
        img2: Second image.
        max_val: The maximum magnitude that a or b can have.
        filter_size: Window size (>= 1).
        filter_sigma: The bandwidth of the Gaussian used for filtering (> 0.).
        k1: One of the SSIM dampening parameters (> 0.).
        k2: One of the SSIM dampening parameters (> 0.).

    Returns:
        Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """

    # Create a 2D Gaussian Kernel
    hw = filter_size // 2
    f_i = ((jnp.arange(filter_size) - hw) / filter_sigma) ** 2
    filt = jnp.exp(-0.5 * f_i)
    filt = filt / jnp.sum(filt)

    kernel = jnp.outer(filt, filt)[:, :, jnp.newaxis, jnp.newaxis]
    kernel = jnp.tile(
        kernel, (1, 1, 1, 3)
    )  # [kernel_height, kernel_width, in_channels, out_channels]

    # This function applies the same 2D filter to every channel of every image in the batch.
    def convolve_2d(img):
        # Transpose the image from [H, W, C] to [C, H, W]
        # because lax.conv_general_dilated expects channels first.
        img_chw = jnp.transpose(img, (2, 0, 1))[jnp.newaxis, ...]

        # feature_group_count=img.shape[-1] makes this a depthwise convolution,
        # applying the (H, W, 1, 3) filter to each of the C channels independently.
        conv_result = jax.lax.conv_general_dilated(
            lhs=img_chw,
            rhs=kernel,
            window_strides=(1, 1),
            padding="VALID",
            feature_group_count=img.shape[-1],  # Depthwise convolution
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )

        # Transpose back to [H', W', C] (HWC)
        return jnp.transpose(conv_result[0], (1, 2, 0))

    mu_a = convolve_2d(img1)
    mu_b = convolve_2d(img2)

    sigma_a_sq = convolve_2d(img1**2) - mu_a**2
    sigma_b_sq = convolve_2d(img2**2) - mu_b**2
    sigma_ab = convolve_2d(img1 * img2) - mu_a * mu_b

    # Compute the SSIM formula
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a_sq + sigma_b_sq + c2)

    ssim_map = numerator / denominator

    return jnp.mean(ssim_map, axis=[0, 1, 2])
