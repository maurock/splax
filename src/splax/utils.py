"""Utility functions"""

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
