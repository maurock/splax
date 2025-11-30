"""Define camera parameters dataclass."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
from flax import struct

from splax import gaussians

Float = jt.Float
Array = jt.Array


@struct.dataclass
class CameraParams:
    """Defines single camera"""

    w2c: Float[Array, "4 4"]
    c2w: Float[Array, "4 4"]
    img_width: int = struct.field(pytree_node=False)
    img_height: int = struct.field(pytree_node=False)
    focal_x: float = struct.field(pytree_node=False)
    focal_y: float = struct.field(pytree_node=False)

    @classmethod
    def make_camera_params(
        cls,
        w2c: Float[Array, "4 4"],
        focal_x: float,
        focal_y: float,
        img_width: int,
        img_height: int,
    ) -> "CameraParams":
        """Create CameraParams with given parameters.
        We assume +Z is forward, +X is right, +Y is up.
        """
        c2w = jnp.linalg.inv(w2c)

        return CameraParams(
            w2c=w2c,
            c2w=c2w,
            focal_x=focal_x,
            focal_y=focal_y,
            img_width=img_width,
            img_height=img_height,
        )


def get_projective_jacobian(
    xyz: Float[Array, "3"],
    camera_params: CameraParams,
) -> tuple[Float[Array, "2 3"], Float[Array, "3 3"], Float[Array, "3"]]:
    """
    Calculates the Jacobian of the perspective projection.
    This approximates the non-linear projection with an affine one locally.

    Jacobian J calculation (from the paper)
    [ dx/dX, dx/dY, dx/dZ ]
    [ dy/dX, dy/dY, dy/dZ ]

    Coordinate mappings:
    u = (x/z)*fx + cx
    J = [ fx/z,  0,   -(fx*x)/z^2 ]
        [ 0,     fy/z, -(fy*y)/z^2 ]
    """
    # Homogeneous coordinates
    xyz_hom = jnp.concatenate([xyz, jnp.ones((1,), dtype=xyz.dtype)])
    xyz_cam = (camera_params.w2c @ xyz_hom)[:3]  # (3)

    x, y, z = xyz_cam[0], xyz_cam[1], xyz_cam[2]
    z = jnp.clip(z, a_min=0.01)

    zeros = jnp.zeros_like(x)

    J = jnp.array(
        [
            [camera_params.focal_x / z, zeros, -(camera_params.focal_x * x) / (z**2)],
            [zeros, camera_params.focal_y / z, -(camera_params.focal_y * y) / (z**2)],
        ]
    )

    w_rot = camera_params.w2c[:3, :3]  # (3, 3)

    return J, w_rot, xyz_cam


def project_gaussians(
    gaussian_params: gaussians.GaussianParams, camera_params: CameraParams
) -> tuple[Float[Array, "2"], Float[Array, "2 2"], Float[Array, ""]]:
    """
    Projection pipeline to extract 2D Means and 2D Covariances
    """
    cov3d = gaussians.compute_covariance_3d(
        gaussian_params.log_scale, gaussian_params.quaternion
    )  # (3, 3)
    J, w_rot, means_cam = get_projective_jacobian(gaussian_params.xyz, camera_params)

    # Important! Stop gradient on J to improve stability
    J = jax.lax.stop_gradient(J)

    # Project covariance Sigma' = J * W * Sigma * W^T * J^T
    T = J @ w_rot
    cov2d = T @ cov3d @ T.transpose(1, 0)  # (2, 2)

    # Project Means
    # u = (x/z)*fx + cx, v = (y/z)*fy + cy
    x, y, z = means_cam[0], means_cam[1], means_cam[2]
    z = jnp.clip(z, a_min=0.001)

    means2d = jnp.array(
        [
            (x / z) * camera_params.focal_x + camera_params.img_width / 2.0,
            (y / z) * camera_params.focal_y + camera_params.img_height / 2.0,
        ]
    )

    return means2d, cov2d, z
